import torch
import numpy as np
from torch.nn.modules.loss import _Loss

class weighted_binary_cross_entropy(_Loss):
    def __init__(self, weights=[1,1], step_size=0.9999, step_per_iter=1, last_weight=1, loss_resolution=None, size_average=None, reduce=None, reduction: str = 'mean', ):
        
        super(weighted_binary_cross_entropy, self).__init__(size_average, reduce, reduction)
        
        for key in weights:
            weights[key]=np.array(weights[key])
            last_weight[key]=np.array(last_weight[key])
            
        self.weights=weights
        self.step_size=np.array(step_size)
        self.step_per_iter = step_per_iter
        self.last_weight = last_weight
        self.now_step=0


        if loss_resolution is None:
            self.loss_resolution_list=[azi for azi in weights.keys()]
        else:
            self.loss_resolution_list=loss_resolution

        

    
    def forward(self, output_dict, target_dict,  mode='train'):

        
        
        loss_dict={}
        loss_batch_mean=0
        loss_each_batch=0

        if self.weights is not None:

            
            
            for azi in target_dict.keys():

                
                
                if azi not in self.weights:
                    print('Weight does not have {} key'.format(azi))

                if azi not in self.loss_resolution_list:
                    continue

                target=target_dict[azi]
                output=output_dict[azi]
            
                loss = self.weights[azi][0] * (target * torch.log(output)) + \
                    self.weights[azi][1] * ((1 - target) * torch.log(1 - output))

                loss=torch.neg(loss)
                loss_mean=loss.mean()
                loss_dict[azi]=loss_mean
                loss_batch_mean+=loss_mean
                loss_each_batch+=loss.mean(dim=(1,2))
                
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        if mode == 'train':
            self.step() 

        return loss_dict, loss_batch_mean, loss_each_batch
    
    def step(self):
        # check whether step is full
        self.now_step+=1
        if self.now_step<self.step_per_iter:
            return


        self.now_step=0
        for key in self.weights.keys():
            new_weight=self.weights[key]*self.step_size

            self.weights[key]=np.clip(new_weight, self.last_weight[key], self.weights[key])

        return 

        


        
    