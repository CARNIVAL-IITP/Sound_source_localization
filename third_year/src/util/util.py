import yaml
import torch
import numpy as np
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import select
import sys
from time import sleep
from signal import signal, alarm, SIGALRM

def load_yaml(yaml_dir):
    yaml_file=open(yaml_dir, 'r')
    data=yaml.safe_load(yaml_file)
    yaml_file.close()
    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def copy_folder(dir):
    shutil.copytree('./',dir+'src', ignore=shutil.ignore_patterns('./wandb/**','*.png', '*.wav'))





def get_yaml_args(yaml_list):
    # print(yaml_list)
    # exit()
    yaml_out={}
    for a in yaml_list:
        a=a.split(' ') 
        yaml_out[a[0]]=load_yaml(a[1])

    
    return yaml_out

def randomseed_init(num):
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num)
        return 'cuda'
    else:
        return 'cpu'

def draw_result_pic(dir, epoch, train, val):
    fig1 = plt.figure(figsize=(7,4))
    epo = np.arange(epoch+1)
   
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    a2 = fig1.add_subplot(1, 1, 1)
    a2.plot(epo, train, epo,val)
    a2.set_title("Loss")
    a2.legend(['Train', 'Eval'])
    a2.set_ylabel('Loss')
    a2.set_xlabel('Epochs')
    a2.grid(axis='y', linestyle='dashed')
    fig1.tight_layout()
    fig1.savefig(dir, dpi=300)
    plt.close(fig1)

def check_list(config):


    
    print('Enter anything for checklist')

    i, o, e = select.select( [sys.stdin], [], [], 10 )

    if (i):
        print('\nChecklist Starts!!')
    else:
        print("\nAuto start!!!")
        return

    print('\nBatch size')
    print(config['train']['dataloader']['batch_size'])

    _=input()

    print('\nnum_workers')
    print(config['train']['dataloader']['num_workers'])
    _=input()

    print('\nGPU ID')
    print(config['train']['GPGPU']['device_ids'])
    _=input()

    print('\nExperiment name')
    print(config['exp']['name'])
    _=input()

    print('\nModel name')
    print(config['model']['name'])
    _=input()
    
    print('\nIs this temp or not? (temp: Y, not: N)')
    temp=input()
    
    if temp in ['N', 'n']:
        config['exp']['temp']=False
    else:
        print('This is temporaray exp!!!')
        config['exp']['temp']=True



def log_saving(record_dir, record_param, epoch, model, writer, optimizer, result_text,restart_num, end=False, temp=False):
    write_file=open(result_text, 'a')
    # print(epoch)
    print("\nAccuracy(train, eval) : %3.3f %3.3f" % (
        record_param['accu_list_train'][epoch] * 100, record_param['accu_list_eval'][epoch] * 100))
    write_file.write("\nAccuracy(train, eval) : %3.3f %3.3f \n" % (
        record_param['accu_list_train'][epoch] * 100, record_param['accu_list_eval'][epoch] * 100))

    print("Max accuracy(eval) : %d epoch, %3.3f\n" % (
        record_param['accu_list_eval'].index(max(record_param['accu_list_eval'])) ,
        max(record_param['accu_list_eval']) * 100))
    write_file.write("Max accuracy(eval) : %d epoch, %3.3f\n\n" % (
        record_param['accu_list_eval'].index(max(record_param['accu_list_eval'])),
        max(record_param['accu_list_eval']) * 100))

    print("Loss(train, eval) : %3.3f %3.3f" % (
        record_param['loss_list_train'][epoch], record_param['loss_list_eval'][epoch]))
    write_file.write("Loss(train, eval) : %3.3f %3.3f\n" % (
        record_param['loss_list_train'][epoch], record_param['loss_list_eval'][epoch]))

    print("Min loss(eval): %d epoch, %3.3f\n" % (
        record_param['loss_list_eval'].index(min(record_param['loss_list_eval'])) , min(record_param['loss_list_eval'])))
    write_file.write("Min loss(eval): %d epoch, %3.3f\n\n" % (
        record_param['loss_list_eval'].index(min(record_param['loss_list_eval'])) , min(record_param['loss_list_eval'])))

    write_file.close()

    if temp ==False:
        writer.add_scalars('Accuracy', {'Accuracy/Train': record_param['accu_list_train'][-1], 'Accuracy/Val':record_param['accu_list_eval'][-1]}, epoch)
        writer.add_scalars('Loss', {'Loss/Train': record_param['loss_list_train'][-1], 'Loss/Val': record_param['loss_list_eval'][-1]}, epoch)

        if (epoch % 10 == 0) or end ==True or epoch<10 or epoch!=0:
            for name, parameter in model.named_parameters():
                writer.add_histogram(name, parameter.clone().detach().cpu().data.numpy(), epoch)

        for x in restart_num:
            if epoch>=(x-6) and epoch<=(x-1):
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'param': record_param
                            }, record_dir + "/model_"+str(epoch)+".pth")

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': record_param
                    }, record_dir + "/current_model.pth")
        if record_param['loss_list_eval'][-1]==min(record_param['loss_list_eval']):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'param': record_param
                        }, record_dir + "/best_loss_model.pth")

        if record_param['accu_list_eval'][-1]==max(record_param['accu_list_eval']):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'param': record_param
                        }, record_dir + "/best_accu_model.pth")
       

        fig1 = plt.figure(figsize=(7,4))
        epo = np.arange(0, epoch + 1, 1)
        a1 = fig1.add_subplot(2, 1, 1)
        a1.plot(epo, np.array(record_param['accu_list_train']), epo, np.array(record_param['accu_list_eval']))
        a1.set_title("Accuracy")
        a1.legend(['Train', 'Eval'])
        a1.set_xlabel('Epochs')
        a1.set_ylabel('Accuracy')
        a1.grid(axis='y', linestyle='dashed')

        a2 = fig1.add_subplot(2, 1, 2)
        a2.plot(epo, np.array(record_param['loss_list_train']), epo, np.array(record_param['loss_list_eval']))
        a2.set_title("Loss")
        a2.legend(['Train', 'Eval'])
        a2.set_ylabel('Loss')
        a2.set_xlabel('Epochs')
        a2.grid(axis='y', linestyle='dashed')
        fig1.tight_layout()
        fig1.savefig(record_dir + '/accu_loss.png', dpi=300)
        plt.close(fig1)
        fig1 = plt.figure(figsize=(7,4))
        epo = np.arange(0, epoch + 1, 1)
        a1 = fig1.add_subplot(2, 1, 1)
        a1.plot(epo, np.array(record_param['accu_list_train']), epo, np.array(record_param['accu_list_eval']))
        a1.set_title("Accuracy")
        a1.legend(['Train', 'Eval'])
        a1.set_xlabel('Epochs')
        a1.set_ylabel('Accuracy')
        a1.grid(axis='y', linestyle='dashed')

        a2 = fig1.add_subplot(2, 1, 2)
        a2.plot(epo, np.array(record_param['loss_list_train']), epo, np.array(record_param['loss_list_eval']))
        a2.set_title("Loss")
        a2.legend(['Train', 'Eval'])
        a2.set_ylabel('Loss')
        a2.set_xlabel('Epochs')
        a2.grid(axis='y', linestyle='dashed')
        fig1.tight_layout()
        fig1.savefig(record_dir + '/accu_loss.png', dpi=300)
        plt.close(fig1)

        
