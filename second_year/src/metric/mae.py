import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from scipy.special import softmax

def get_pw_losses(loss_func, est_targets, targets, **kwargs):
       
    batch_size, n_src, *_ = targets.shape
    pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
    for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
        for target_idx, target_src in enumerate(targets.transpose(0, 1)):
            pair_wise_losses[:, est_idx, target_idx] = loss_func(est_src, target_src, **kwargs)

    

    pair_wise_losses=pair_wise_losses.numpy()

    each_batch_error=np.zeros((batch_size, n_src))

    for b in range(batch_size):
        row_ind, col_ind =linear_sum_assignment(pair_wise_losses[b])
        
        result=pair_wise_losses[b][row_ind, col_ind]
        each_batch_error[b]=result
   
    return each_batch_error

def get_doa_error(est_targets, targets):

    doa_error= torch.abs(est_targets-targets)
    doa_error_360=torch.abs(360-doa_error)
    

    cat=torch.stack([doa_error, doa_error_360], dim=1)   
    
    
    cat=torch.min(cat, dim=1).values.float()


    return cat

def find_max_point(maximum_distance, num_spk, data, ):
    softmax_data=deepcopy(data)
    
    argmax_list=np.zeros(num_spk)
    softmax_list=np.zeros(num_spk)
    softmax_half_list=np.zeros(num_spk)
 
    for j in range(num_spk):

        current_max=data.argmax()

        argmax_list[j]=current_max

        near_degree=np.arange(-maximum_distance, maximum_distance+1)+current_max

        delete_degree_index=near_degree%360
        near_degree_value=softmax_data[delete_degree_index]


        half_near_degree=near_degree[maximum_distance//2:-maximum_distance//2]        
        half_delete_degree_index=delete_degree_index[maximum_distance//2:-maximum_distance//2]
      
        half_near_degree_value=softmax_data[half_delete_degree_index]

        
        near_degree_value= -np.log((1-near_degree_value)/(near_degree_value+1e-8)) # inverse sigmoid
        near_degree_value=softmax(near_degree_value)
        
        near_degree=near_degree*near_degree_value
        softmax_list[j]=near_degree.sum()%360

        half_near_degree_value=-np.log((1-half_near_degree_value)/(half_near_degree_value+1e-8)) # inverse sigmoid

        half_near_degree_value=softmax(half_near_degree_value)

        half_near_degree_value=half_near_degree*half_near_degree_value
        softmax_half_list[j]=half_near_degree_value.sum()%360
        
        data[delete_degree_index]=0

    return argmax_list, softmax_list, softmax_half_list



def calc_mae(output, target, vad, num_spk, azimuth, resolution=1, local_maximum_distance=10, calc_layer=[0], acc_threshold=5, ref_vec=0):

    number_of_degrees_to_estimate=0

    total_argmax_acc=0
    total_softmax_acc=0

    total_softmax_doa_error=0
    total_argmax_doa_error=0

    total_half_softmax_acc=0

    total_half_softmax_doa_error=0


    
    
    output=output.numpy()
    azimuth=azimuth.repeat_interleave(len(calc_layer), 0)
    

    for frame_num in range(output.shape[-1]):

        vad_frame=vad[..., frame_num] # B, num_spk

        if vad_frame.any()==0: # no speech in the frame
            continue

        active_spk=np.where(vad_frame[0]==1)[0]    
        active_spk_num=active_spk.shape[0]   
        number_of_degrees_to_estimate+=active_spk_num

        frame_azimuth=azimuth[:, active_spk]
        
        
        
        out_frame=output[..., frame_num] # B, 360//resolution

        argmax_peak_list=[]
        softmax_peak_list=[]
        half_softmax_peak_list=[]

        for layer in calc_layer:
            
            

            local_argmax_point, local_softmax_point, local_half_softmax_point=find_max_point(local_maximum_distance, active_spk_num, out_frame[0, layer])
            
            
            argmax_peak_list.append(local_argmax_point)
            softmax_peak_list.append(local_softmax_point)
            half_softmax_peak_list.append(local_half_softmax_point)
            
        argmax_estimate_frame=np.stack(argmax_peak_list) 
        softmax_estimate_frame=np.stack(softmax_peak_list)
        half_softmax_estimate_frame=np.stack(half_softmax_peak_list)
        
        estimate_frame=np.concatenate((argmax_estimate_frame, softmax_estimate_frame, half_softmax_estimate_frame))
   
        estimate_frame=torch.from_numpy(estimate_frame) +ref_vec     
        

        estimate_frame=estimate_frame%360
 

        frame_azimuth=frame_azimuth.repeat_interleave(3,dim=0)

        
        doa_error=get_pw_losses(get_doa_error, estimate_frame, frame_azimuth)   

     
        acc=doa_error<=acc_threshold

        acc=acc.astype(int).mean(axis=1)
        doa_error=doa_error.sum(axis=-1)
        

       
        
        total_argmax_acc=total_argmax_acc+acc[:len(calc_layer)]
        total_softmax_acc=total_softmax_acc+acc[len(calc_layer):2*len(calc_layer)]
        total_half_softmax_acc=total_half_softmax_acc+acc[-len(calc_layer):]
        

        total_argmax_doa_error=total_argmax_doa_error+doa_error[:len(calc_layer)]
        total_softmax_doa_error=total_softmax_doa_error+doa_error[len(calc_layer):2*len(calc_layer)]
        total_half_softmax_doa_error=total_half_softmax_doa_error+doa_error[-len(calc_layer):]


   
    return total_argmax_acc, total_softmax_acc,total_half_softmax_acc, total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error, number_of_degrees_to_estimate
