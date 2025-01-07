from .FFT import EPSILON, ConvSTFT
from torch import nn
import torch
import numpy as np

class Causal_Conv2D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Causal_Conv2D_Block, self).__init__()
        
        
        self.conv2d=nn.Conv2d(*args, **kwargs)


        self.norm=nn.BatchNorm2d(args[1])

        
        self.activation=nn.ELU()

    def forward(self, x):
        original_frame_num=x.shape[-1]
        x=self.conv2d(x)
        x=self.norm(x)
        x=self.activation(x)   
        x=x[...,:original_frame_num]    
        
       

        return x

class Conv1D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1D_Block, self).__init__()
        
        
        self.conv1d=nn.Conv1d(*args, **kwargs)


        self.norm=nn.BatchNorm1d(args[1])
        
        self.activation=nn.ELU()

    def forward(self, x):
        
        x=self.conv1d(x)
        x=self.norm(x)
        x=self.activation(x)       
        
       

        return x


class crn(nn.Module):
    def __init__(self, config, output_num, azi_size):
        super(crn, self).__init__()

        
        self.output_num=output_num
        self.azi_size=azi_size

        
        self.cnn_num=config['CNN']['layer_num']
        self.kernel_size=config['CNN']['kernel_size']
        self.filter_size=config['CNN']['filter']

        self.max_pool_kernel=config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=config['CNN']['max_pool']['stride']

        args=[2*(config['input_audio_channel']-1),self.filter_size,self.kernel_size] # in_channel, out_channel, kernel size
       
        kwargs={'stride': 1, 'padding': [1,2], 'dilation': 1}

      


        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))
        self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))

        args[0]=config['CNN']['filter']       
        for count in range(self.cnn_num-1):
            self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))
            self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))
    
        self.GRU_layer=nn.GRU(**config['GRU'])
        self.h0=torch.zeros(*config['GRU_init']['shape'])
        self.h0=torch.nn.parameter.Parameter(self.h0, requires_grad=config['GRU_init']['learnable'])
        

        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()

        args[0]=config['GRU']['hidden_size']
        args[1]=config['GRU']['hidden_size']
        args[2]=1
        kwargs['padding']=0
      
        for _ in range(output_num):
            self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
            self.azi_mapping_final.append(nn.Conv1d(config['GRU']['hidden_size'], self.azi_size, 1))
     
        

        
        

    def forward(self, x):
      
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):

            x=cnn_layer(x)[...,:x.shape[-1]]
            x=pooling_layer(x)

        
        
        b, c, f, t=x.shape
        x=x.view(b, -1, t).permute(0,2,1)


        h0=self.h0.repeat_interleave(x.shape[0])
        self.GRU_layer.flatten_parameters()
        
        h0=h0.view(self.h0.shape[0], x.shape[0], self.h0.shape[-1])

        x, h=self.GRU_layer(x, h0)

        x=x.permute(0,2,1)
        
        outputs=[]

        for final_layer, cnn_layer in zip(self.azi_mapping_final, self.azi_mapping_conv_layer):
            x=cnn_layer(x)
            res_output=final_layer(x)
            outputs.append(res_output)
        output=torch.stack(outputs).permute(1,0,2,3)
        
        return output


class main_model(nn.Module):
    def __init__(self, config):
        super(main_model, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps
        self.ref_ch=self.config['ref_ch']

        self.stft_model=ConvSTFT(**self.config['FFT'])
        self.crn=crn(self.config['CRN'], 3, 360)
    



    def irtf_featue(self, x, target):
        r, i, target =self.stft_model(x, target, cplx=True)
       
        comp = torch.complex(r, i)
        
        comp_ref = comp[..., [self.ref_ch], :, :]
        comp_ref = torch.complex(
        comp_ref.real.clamp(self.eps), comp_ref.imag.clamp(self.eps)
        )

        comp=torch.cat(
        (comp[..., self.ref_ch-1:self.ref_ch, :, :], comp[..., self.ref_ch+1:, :, :]),
        dim=-3) / comp_ref
        x=torch.cat((comp.real, comp.imag), dim=1)

        return x, target
    

    


        
    def forward(self, x, vad):

    
        x, vad_frame=self.irtf_featue(x, vad)
        
    
        x=self.crn(x)
        
        return x, vad_frame



