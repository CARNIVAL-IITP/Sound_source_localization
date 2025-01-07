import sys, os

import util
import torch

import numpy as np
import random
import importlib

from tqdm import tqdm
from dataloader.data_loader import IITP_test_dataload
import matplotlib.pyplot as plt
import pandas as pd

import metric


class Hyparam_set():
    
    def __init__(self, args):
        self.args=args
    

    def set_torch_method(self,):
        try:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=False) # spawn
        except:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=True) # spawn
        

    def randomseed_init(self,):
        np.random.seed(self.args['hyparam']['randomseed'])
        random.seed(self.args['hyparam']['randomseed'])
        torch.manual_seed(self.args['hyparam']['randomseed'])
        # torch.Generator.manual_seed(self.args['randomseed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
        self.args['hyparam']['GPGPU']['device']=device
        return device
    
    def set_on(self):
        self.set_torch_method()
        self.device=self.randomseed_init()
       
        return self.args

class Learner_config():
    def __init__(self, args) -> None:
        self.args=args
    



    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        
        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model(self.args['model']).to(self.device)

        trained=torch.load(self.args['hyparam']['model'], map_location=self.device)
        self.model.load_state_dict(trained['model_state_dict'], )
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
        

   
  

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()
   
        
        return self.args

class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']
        
        
        
       


    def save_output(self, DB_type):
        try:
            now_dict=self.save_config_dict[DB_type]
        except:
            now_dict=self.save_config_dict[int(DB_type)]
            DB_type=int(DB_type)

        
        
        now_file_csv=self.mae_per_room[DB_type]
        pd.DataFrame(now_file_csv).to_csv(self.result_folder['inference_folder']+'/'+DB_type+'/result.csv', index=True)


        with open(self.result_folder['inference_folder']+'/'+DB_type+'/result.txt', 'w') as f:
            f.write('\nargmax_acc\n\n')

            k=(now_dict['argmax_acc']/now_dict['number_of_degrees'])
            # print(k)
            # exit()
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nargmax_doa_error\n\n')
            k=(now_dict['argmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
            
            
          
            f.write('\n\n')

            f.write('\nsoftmax_acc\n\n')

            k=(now_dict['softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nsoftmax_doa_error\n\n')
            k=(now_dict['softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
        
            f.write('\n\n')


            f.write('\nhalf_softmax_acc\n\n')

            k=(now_dict['half_softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nhalf_softmax_doa_error\n\n')
            k=(now_dict['half_softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
    def plotting_target_output(self, iter_num, out, target):
        out=out.sigmoid().detach().cpu().numpy()[0]
        target=target.detach().cpu().numpy()[0]
        azi_resolution=1
        
        #### plotting
        if iter_num%200==0:
            # print(self.loss_function.weights[1])
            
            fig=plt.figure(figsize=(7,7),)#, nrows=azi_num, ncols=1, sharey=True)

            # for (row, big_ax), azi  in zip(enumerate(big_axes, start=1),output_dict.keys()):
            #     big_ax.set_title(azi, fontsize=16)
            
          
            
            for num in range(out.shape[0]):
                
                output_now=out[num]
                target_now=target[num]

                

                
                # lj=plt.imshow(target_now,  vmin=0, vmax=1.0, cmap = plt.get_cmap('plasma'), aspect='auto')
                # plt.colorbar(lj)
                # plt.savefig('../results/estimate/png_sample.png')
                # exit()
                
                
              

                ytick_front=[0,output_now.shape[0]//2, output_now.shape[0]-1]
                ytick_back=[0,(output_now.shape[0]//2)*azi_resolution, (output_now.shape[0]-1)*azi_resolution]

                plt.subplot(out.shape[0],3,num*3+1)
                
                
                lj=plt.imshow(output_now,  vmin=0, vmax=1.0, cmap = plt.get_cmap('plasma'), aspect='auto')
                plt.colorbar(lj)
                
                plt.yticks(ytick_front, ytick_back)
                # plt.title('{} Output'.format(azi))
                
                # plt.imshow(target_now)
                # plt.savefig('../results/estimate/png_sample.png')
                # exit()
                plt.subplot(out.shape[0],3,num*3+2)
                

                output_gap=target_now-output_now
                lj=plt.imshow(output_gap,vmin=-1.0, vmax=1.0,  cmap = plt.get_cmap('seismic'), aspect='auto')
                plt.colorbar(lj)
                plt.yticks(ytick_front, ytick_back)
                


                plt.subplot(out.shape[0],3,num*3+3)
                
                
                tk=plt.imshow(target_now , vmin=0, vmax=1, cmap = plt.get_cmap('plasma'),  aspect='auto')
                plt.colorbar(tk)
                plt.yticks(ytick_front, ytick_back)
                
            png_name='../results/estimate/estimaste_{}.png'.format(iter_num)
            os.makedirs(os.path.dirname(png_name), exist_ok=True)
            plt.tight_layout()
            plt.savefig(png_name, dpi=400,)
            plt.close()
            plt.clf()
            plt.cla()
            # exit()
            
        
        return 
    

    
    def error_update(self, DB_type, argmax_acc, softmax_acc, half_softmax_acc,argmax_doa_error, softmax_doa_error, half_softmax_doa_error,num, iter_num):
        
        self.mae_per_room[DB_type].append(argmax_doa_error/num)
        # print(argmax_doa_error/num)
        # exit()

        now_dict=self.save_config_dict[DB_type]
        
        now_dict['argmax_acc']+=argmax_acc
        now_dict['softmax_acc']+=softmax_acc
        now_dict['half_softmax_acc']+=half_softmax_acc
     
        now_dict['argmax_doa_error']+=argmax_doa_error
        now_dict['softmax_doa_error']+=softmax_doa_error
        now_dict['half_softmax_doa_error']+=half_softmax_doa_error

        now_dict['number_of_degrees']+=num
        self.save_config_dict[DB_type]=now_dict
        # print(self.save_config_dict[DB_type])
        # exit()

    def config(self,):
        from copy import deepcopy

        self.save_config_dict=dict()

        metric_data={}
        metric_data['argmax_acc']=0
        metric_data['argmax_doa_error']=0
        

        metric_data['softmax_acc']=0
        metric_data['softmax_doa_error']=0

        metric_data['half_softmax_acc']=0
        metric_data['half_softmax_doa_error']=0



        metric_data['number_of_degrees']=0

        self.mae_per_room=dict()
        
        for room_type in self.result_folder['room_type']:
            os.makedirs(self.result_folder['inference_folder']+room_type, exist_ok=True)

            
            self.save_config_dict[room_type]=deepcopy(metric_data)
            self.mae_per_room[room_type]=[]

   

        return self.args

       



        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        self.test_loader=IITP_test_dataload(self.args['dataloader']['test'])
        
       
        return self.args
        
        

class Tester():

    def __init__(self, args):

        # self.temp()
        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()
        self.model=self.learner.model


        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()
        

    
    def run(self, ):   
        
        self.test()   
        
        

    def test(self, ):
        self.model.eval()

        doa_error_list=[]



        mic_type=self.args['dataloader']['test']['mic_type']

        audio_save_dir='../results/ellipsoid_6_result/'
        os.makedirs(audio_save_dir, exist_ok=True)


        for room_type in tqdm(self.args['hyparam']['result_folder']['room_type'], desc='room, mic_type: '+mic_type):
            room_type=str(room_type)
      
            self.dataloader.test_loader.dataset.room_type=str(room_type)
           
            self.dataloader.test_loader.dataset.pkl_list=os.listdir(self.dataloader.test_loader.dataset.pkl_dir+room_type)
            with torch.no_grad():
                for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(self.dataloader.test_loader):
                    # continue

                    
                   
                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)

                    out, vad=self.model(mixed, vad)

                 

                    out=out.sigmoid().detach().cpu()

                    vad=vad.cpu()
                    speech_azi=speech_azi.cpu()

                    if self.dataloader.test_loader.dataset.args['mic_type']=='linear' or self.dataloader.test_loader.dataset.args['mic_type']=='whole':
                    

                        out[..., 180:,:]=0



        
                    


                    total_argmax_doa_error, number_of_degrees_to_estimate=metric.mae.calc_mae(out, vad,  speech_azi, calc_layer=[2],\
                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])

                    
                    doa_error_list.append(total_argmax_doa_error/number_of_degrees_to_estimate)

                    import soundfile as sf

                    audio_name=audio_save_dir+room_type+'/'+str(iter_num)+'/'
                    os.makedirs(audio_name, exist_ok=True)

                    mixed=mixed.cpu().numpy()[0][0]
                    sf.write(audio_name+'mixed.wav', mixed, 16000)
                    out=out.numpy()[0][-1]
                    plt.imshow(out, aspect='auto')#, vmax=1.0, vmin=0.0)
                    plt.savefig(audio_name+'out.png')
                    plt.close()
                    plt.clf()
                    plt.cla()


                    

                    self.learner.memory_delete([mixed, vad, speech_azi, out, ])

        df=pd.DataFrame(doa_error_list)
        df.to_csv('../results/ellipsoid_6_result.csv', index=False)

        




if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)

    t=Tester(args)
    t.run()