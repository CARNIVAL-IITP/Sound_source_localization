import sys, os
import util
import torch
import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.data_loader_for_db_make import Test_data_maker_load, val_data_maker_load
import matplotlib.pyplot as plt
import pandas as pd

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
        
        self.model=None
        

    def init_optimizer(self):
        
        a=importlib.import_module('torch.optim')
        assert hasattr(a, self.args['learner']['optimizer']['type']), "optimizer {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer']['type'])
     
        self.optimizer=a(self.model.parameters(), **self.args['learner']['optimizer']['config'])
        self.gradient_clip=self.args['learner']['optimizer']['gradient_clip']
    
        
    def init_optimzer_scheduler(self, ):
        a=importlib.import_module('torch.optim.lr_scheduler')
        assert hasattr(a, self.args['learner']['optimizer_scheduler']['type']), "optimizer scheduler {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer_scheduler']['type'])

        self.optimizer_scheduler=a(self.optimizer, **self.args['learner']['optimizer_scheduler']['config'])



    def init_loss_func(self):
        
        

        if self.args['learner']['loss']['type']=='weighted_bce':
            from loss.bce_loss import weighted_binary_cross_entropy
            self.loss_func=weighted_binary_cross_entropy(**self.args['learner']['loss']['option'])
        elif self.args['learner']['loss']['type']=='BCEWithLogitsLoss':
            self.loss_func=torch.nn.modules.loss.BCEWithLogitsLoss(reduction='none')
            self.loss_func=torch.nn.modules.loss.BCELoss(reduction='none')


        elif self.args['learner']['loss']['type']=='kld':
            self.loss_func=torch.nn.modules.loss.KLDivLoss(reduction='none')
        elif self.args['learner']['loss']['type']=='mse':
            self.loss_func=torch.nn.modules.loss.MSELoss(reduction='none')

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']
        self.loss_weight=self.args['learner']['loss']['option']['each_layer_weight']

        if self.args['learner']['loss']['optimize_method']=='min':
            self.best_val_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_val_loss=-math.inf
            self.best_train_loss=-math.inf

    def train_update(self, output, target):
        target=target[:, self.loss_train_map_num]
        output=output[:, self.loss_train_map_num].sigmoid()
        loss=self.loss_func(output, target)

        for j in range(len(self.loss_weight)):
            loss[:, j]=loss[:,j]*self.loss_weight[j]



        loss_mean=loss.mean()
       

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_mean

    def test_update(self, output, target):
       

        target=target[:, self.loss_train_map_num]
        output=output[:, self.loss_train_map_num].sigmoid()
        loss=self.loss_func(output, target)

        for j in range(len(self.loss_weight)):
            loss[:, j]=loss[:,j]*self.loss_weight[j]
        loss_mean=loss.mean()
        

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        return loss_mean

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']

        return self.args

class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.csv=dict()
        self.csv['train_epoch_loss']=[]
        self.csv['train_best_loss']=[]
        self.csv['test_epoch_loss']=[]
        self.csv['test_best_loss']=[]

        self.csv_dir=self.args['logger']['save_csv']
        self.model_save_dir=self.args['logger']['model_save_dir']
        self.png_dir=self.args['logger']['png_dir']

        if self.args['logger']['optimize_method']=='min':
            self.best_test_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_test_loss=-math.inf
            self.best_train_loss=-math.inf

    def train_iter_log(self, loss):
        try:
            wandb.log({'train_iter_loss':loss})
        except:
            None
        self.epoch_train_loss.append(loss.cpu().detach().item())

       
 
    def train_epoch_log(self):
        loss_mean=np.array(self.epoch_train_loss).mean()

        self.csv['train_epoch_loss'].append(loss_mean)

        if self.best_train_loss > loss_mean:
            self.best_train_loss = loss_mean 

        try:
            wandb.log({'train_epoch_loss':loss_mean})
            wandb.log({'train_best_loss':self.best_train_loss})
        except:
            None

        self.csv['train_best_loss'].append(self.best_train_loss)



    def test_iter_log(self, loss):
        try:
            wandb.log({'test_iter_loss':loss})
        except:
            None
        self.epoch_test_loss.append(loss.cpu().detach().item())

    def test_epoch_log(self, optimizer_scheduler):
        loss_mean=np.array(self.epoch_test_loss).mean()
        self.csv['test_epoch_loss'].append(loss_mean)

        self.model_save=False
        if self.best_test_loss > loss_mean:
            self.model_save=True
            self.best_test_loss = loss_mean 
        try:
            wandb.log({'test_epoch_loss':loss_mean})
            wandb.log({'test_best_loss':self.best_test_loss})
        except:
            None
        self.csv['test_best_loss'].append(self.best_test_loss)

        optimizer_scheduler.step(loss_mean)
        

        

    def epoch_init(self,):
        self.epoch_train_loss=[]
        self.epoch_test_loss=[]
    

    def epoch_finish(self, epoch, model, optimizer):
        os.makedirs(os.path.dirname(self.csv_dir), exist_ok=True)
        pd.DataFrame(self.csv).to_csv(self.csv_dir)

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
        if self.model_save:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_model.tar")
            print("new best model\n")
        torch.save(checkpoint,  self.model_save_dir + "{}_model.tar".format(epoch))

        
        util.util.draw_result_pic(self.png_dir, epoch, self.csv['train_epoch_loss'],  self.csv['test_epoch_loss'])



    def wandb_config(self):
     
        return self.args  
  
    def config(self,):
        self.wandb_config()
        return self.args
        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    
    def config(self):
        self.test_loader=Test_data_maker_load(self.args['dataloader']['test'])
        self.val_loader=val_data_maker_load(self.args['dataloader']['val'])
      
        return self.args   
        
        
        

class Trainer():

    def __init__(self, args):

        # self.temp()
        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()
     

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()       

     
        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()
        

    
    def run(self, ):
        
        self.validation(0)
       
        self.test(0)

    def validation(self, epoch):

  
        with torch.no_grad():
            for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.val_loader , desc='Test', total=len(self.dataloader.val_loader), )):
                # break
                continue
                
            
            # pkl_csv='./metadata/val_csv_linear_8.csv'
            # pkl_csv='./metadata/val_csv_ellipsoid_6.csv'
            pkl_csv='./metadata/val_csv_circular_4.csv'

            pkl_dir=self.dataloader.val_loader.dataset.pkl_dir
            pkl_list=os.listdir(pkl_dir)
            df={}
            df['data']=pkl_list
            pd.DataFrame(df).to_csv(pkl_csv)



    def test(self, epoch):
   
        with torch.no_grad():
            for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader), )):
                # break
                continue
                
            # pkl_csv='./metadata/test_csv_linear_8.csv'
            # pkl_csv='./metadata/test_csv_ellipsoid_6.csv'
            pkl_csv='./metadata/test_csv_circular_4.csv'

            pkl_dir=self.dataloader.test_loader.dataset.pkl_dir
            pkl_list=os.listdir(pkl_dir)
            df={}
            df['data']=pkl_list
            pd.DataFrame(df).to_csv(pkl_csv)

            




if __name__=='__main__':
    args=sys.argv[1:]
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()