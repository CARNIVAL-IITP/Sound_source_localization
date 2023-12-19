import sys, os

import util
import torch

import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.data_loader import IITP_test_dataload
import matplotlib.pyplot as plt
import pandas as pd
import pickle
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
        

    def init_loss_func(self):
        
        

        if self.args['learner']['loss']['type']=='weighted_bce':
            from loss.bce_loss import weighted_binary_cross_entropy
            self.loss_func=weighted_binary_cross_entropy(**self.args['learner']['loss']['option'])
        elif self.args['learner']['loss']['type']=='BCEWithLogitsLoss':
            self.loss_func=torch.nn.modules.loss.BCEWithLogitsLoss(reduction='none')
            self.loss_func=torch.nn.modules.loss.BCELoss(reduction='none')

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']
    
    def update(self, output, target):
       

        target=target[:, self.loss_train_map_num]
        output=output[:, self.loss_train_map_num].sigmoid()

        loss=self.loss_func(output, target)
    
        
        loss_mean=loss.mean()


        return loss_mean

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()
        self.init_loss_func()
        
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
    

    
    def error_update(self, DB_type, argmax_acc, softmax_acc, half_softmax_acc,argmax_doa_error, softmax_doa_error, half_softmax_doa_error,num):
        now_dict=self.save_config_dict[DB_type]
        
        now_dict['argmax_acc']+=argmax_acc
        now_dict['softmax_acc']+=softmax_acc
        now_dict['half_softmax_acc']+=half_softmax_acc
     
        now_dict['argmax_doa_error']+=argmax_doa_error
        now_dict['softmax_doa_error']+=softmax_doa_error
        now_dict['half_softmax_doa_error']+=half_softmax_doa_error

        now_dict['number_of_degrees']+=num
        self.save_config_dict[DB_type]=now_dict
   

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
        
        for room_type in self.result_folder['room_type']:
            os.makedirs(self.result_folder['inference_folder']+room_type, exist_ok=True)

            
            self.save_config_dict[room_type]=deepcopy(metric_data)

   

        return self.args

       



        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        self.test_loader=IITP_test_dataload(self.args['dataloader']['test'])
        
       
        return self.args
        
        

class Tester():

    def __init__(self, args):

      
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


        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)
            self.dataloader.test_loader.dataset.room_type=str(room_type)
      
            print(room_type)
            print('\n')
            with torch.no_grad():
                for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader), )):
     
                   
                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)

                    out, target, vad=self.model(mixed, vad, speech_azi, iter_num, 0)

                    out=out.sigmoid().detach().cpu()

                    

                    if self.dataloader.test_loader.dataset.args['mic_type']=='linear':
                    
                        out_copy=out.clone()
                        out[:,:,181:,:]=0
                        out_copy[:,:,:181,:]=0
                        out_copy=torch.flip(out_copy, dims=[2])
                        out=torch.stack([out, out_copy], dim=0).max(dim=0).values
           
                    target=target.cpu()
                    vad=vad.cpu()
                    speech_azi=speech_azi.cpu()

          
                    
                    
           
                    

                    total_argmax_acc, total_softmax_acc, total_half_softmax_acc, total_argmax_doa_error, total_softmax_doa_error,total_half_softmax_doa_error, number_of_degrees_to_estimate=metric.mae.calc_mae(out, target, vad, num_spk, speech_azi,\
                        calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])
                 
                    self.logger.error_update(room_type, total_argmax_acc, total_softmax_acc,total_half_softmax_acc, total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error,number_of_degrees_to_estimate)

                    

                    self.learner.memory_delete([mixed, vad, speech_azi, out, target,])
      
                self.logger.save_output(room_type)




if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)

    t=Tester(args)
    t.run()