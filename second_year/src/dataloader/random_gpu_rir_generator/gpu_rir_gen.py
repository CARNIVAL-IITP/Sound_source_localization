import numpy as np
import os
import gpuRIR
import plotly.graph_objects as go
from pathlib import Path
import parmap
from glob import glob
import librosa
from soundfile import write as audiowrite
import scipy.signal as ss
from torch import abs_
from tqdm import tqdm
import shutil
import csv
import time
import multiprocessing as mp
import yaml
import matplotlib.pyplot as plt
import random
import math
import numpy.matlib
import pandas as pd

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)


class simulator_common():
    def __init__ (self,):    
        circle=self.circle_mic_pos()
        ellipsoid=self.ellip_mic_pos()
        linear=self.linear_mic_pos()
        self.mic_pos_dict={'circle':circle, 'ellipsoid':ellipsoid, 'linear':linear}


        mic_list=[]
        for shape in ['circle', 'ellipsoid']:
            for num in [4,6,8]:            
                mic_list.append(self.mic_pos_dict[shape][num])
        mic_list.append(self.mic_pos_dict['linear'][8])

        self.whole_mic_setup={}
        self.whole_mic_setup['arrayType']='2D'
        self.whole_mic_setup['orV'] = np.array([0.0, 1.0, 0.0]),

        self.whole_mic_setup['mic_pos']=np.concatenate(mic_list, axis=0)

        self.whole_mic_setup['mic_orV']=None
        self.whole_mic_setup['mic_patter']='omni'
       
     

    def circle_mic_pos(self):
        pos_4=np.array([[3.231, 0, 0.0],
                        [0, 3.231, 0.0],
                        [-3.231, 0, 0.0],
                        [0, -3.231, 0.0]])
        

        pos_6=np.array([[4.57, 0, 0.0],
                        [2.285, 3.957736095, 0.0],
                        [-2.285, 3.957736095, 0.0],
                        [-4.57,0, 0.0],
                        [-2.285, -3.957736095, 0.0],
                        [2.285, -3.9577360955, 0.0]])

        pos_8=np.array([[5.970992749, 0, 0.0],
                        [4.222129464, 4.222129464, 0.0],
                        [0, 5.970992749, 0.0],
                        [-4.222129464, 4.222129464, 0.0],
                        [-5.970992749, 0, 0.0],
                        [-4.222129464, -4.222129464, 0.0],
                        [0,-5.970992749, 0.0],
                        [4.222129464, -4.222129464, 0.0]])
        return {4:pos_4/100, 6:pos_6/100, 8:pos_8/100}

    def ellip_mic_pos(self):
        pos_4=np.array([[2.4210, 0, 0.0],
                        [0, 3.1841, 0.0],
                        [-2.4210, 0, 0.0],
                        [0, -3.1841, 0.0]])

        pos_6=np.array([[4.6149, 0, 0.0],
                [2, 3.0269, 0.0],
                [-2, 3.2069, 0.0],
                [-4.6149, 0, 0.0],
                [-2, -3.2069, 0.0],
                [2, -3.0269, 0.0]])

       

        pos_8=np.array([[5.9229, 0, 0.0],
                [3.8509, 3.4216, 0.0],
                [0, 4.5033, 0.0],
                [-3.8509, 3.4216, 0.0],
                [-5.9229, 0, 0.0],
                [-3.8509, -3.4216, 0.0],
                [0, -4.5033, 0.0],
                [3.8509, -3.4216, 0.0]])





        return {4:pos_4/100, 6:pos_6/100, 8:pos_8/100}
    
    def linear_mic_pos(self):

        pos_4=np.array([[6, 0, 0],
                        [2, 0, 0],
                        [-2, 0, 0],
                        [-6, 0, 0]])
        
        
        pos_6=np.array([[10, 0, 0],
                        [6, 0, 0],
                        [2, 0, 0],
                        [-2, 0, 0],
                        [-6, 0, 0],
                        [-10, 0, 0]])
        

        pos_8=np.array([[14, 0, 0],
                        [10, 0, 0],
                        [6, 0, 0],
                        [2, 0, 0],
                        [-2, 0, 0],
                        [-6, 0, 0],
                        [-10, 0, 0],
                        [-14, 0, 0]])
        # pos_8=np.flip(pos_8, -1)
        return {4:pos_4/100, 6:pos_6/100, 8:pos_8/100}



    



class acoustic_simulator_on_the_fly(simulator_common):

    def __init__ (self, config):
        super(acoustic_simulator_on_the_fly, self).__init__()

    
        self.fs=config['fs']
        self.rir_character_dict=config['gpu_rir_characteristic']     
       
        
        # initialize_room_param
        self.params=self.rir_character_dict['gpu_rir_generate_dict'] # c, fs, r, s, L, beta, reverberation_time, nsample, mtype, order, dim, orientation, hp_filter
       
        
        

    def get_random_value(self, bound):
        max_value=np.array(bound[1])
        min_value=np.array(bound[0])

        return min_value + np.random.random(min_value.shape) * (max_value - min_value)



    def random_room_select(self):


        room_param={}

        
        room_sz_bound=self.rir_character_dict['room_sz_bound']
        room_sz=self.get_random_value(room_sz_bound)
        room_param['L']=room_sz
        

        rt60=np.random.uniform(*self.rir_character_dict['rt60_bound'])
        room_param['reverberation_time']=rt60
        abs_weight=np.random.uniform(*self.rir_character_dict['abs_weights_bound'], size=6)      
       

        
        

        return room_sz, rt60, abs_weight

    def mic_select(self):
        n_mic=random.choice(self.rir_character_dict['mic']['mic_num']) # number of mic         
        mic_shape=random.choice(self.rir_character_dict['mic']['mic_shape']) # mic shape

        mic_pos=self.mic_pos_dict[mic_shape][n_mic]/100 # cm to m
        return mic_pos, n_mic

    def gpu_rir_param(self, room_sz, rt60, abs_weight,):
        beta = gpuRIR.beta_SabineEstimation(room_sz, rt60, abs_weights=abs_weight)
        
        
        self.params['room_sz']=room_sz    
        self.params['beta']=beta

        if rt60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1,1,1]

        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(12, rt60) # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(40, rt60)  # Use diffuse model until the RIRs decay 40dB
            if rt60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n( Tdiff, room_sz )
        self.params['Tdiff']=Tdiff
        self.params['Tmax']=Tmax
        self.params['nb_img']=nb_img

            



    def get_theta_mic_center(self, ):
        theta=random.uniform(0, 2*math.pi)
        theta=np.rad2deg(theta)

        mic_height=random.uniform(*self.rir_character_dict['mic']['mic_height'])
   
        
        
        mic_loc_x_range=self.params['room_sz'][0]/2-self.rir_character_dict['mic']['mic_from_wall']
        mic_loc_x_range=[-mic_loc_x_range, mic_loc_x_range]
        mic_loc_x=random.uniform(*mic_loc_x_range) + self.params['room_sz'][0]/2

        mic_loc_y_range=self.params['room_sz'][1]/2-self.rir_character_dict['mic']['mic_from_wall']
        mic_loc_y_range=[-mic_loc_y_range, mic_loc_y_range]
        mic_loc_y=random.uniform(*mic_loc_y_range) + self.params['room_sz'][1]/2

        mic_center=np.array([mic_loc_x, mic_loc_y, mic_height])

        return theta, mic_center
    
    def mic_rotate_location(self,mic_pos, n_mic, room_sz, orV_rcv):
        ###### mic rotation

        while True:


            theta=random.uniform(0, 2*math.pi)

            c, s=np.cos(theta), np.sin(theta)      
            R=np.array(((c, -s), (s,c)))
            mic_pos=R.dot(mic_pos[:,:2].T).T      
            theta=np.rad2deg(theta)



            mic_height=random.uniform(*self.rir_character_dict['mic']['mic_height'])
            mic_loc=np.zeros((n_mic, 3), dtype=np.float32)
            mic_loc[:,-1]=mic_height
            
            mic_loc[:, :2]=mic_pos
            
            
            mic_loc_x_range=self.params['room_sz'][0]/2-self.rir_character_dict['mic']['mic_from_wall']
            mic_loc_x_range=[-mic_loc_x_range, mic_loc_x_range]
            mic_loc_x=random.uniform(*mic_loc_x_range) + self.params['room_sz'][0]/2

            mic_loc_y_range=self.params['room_sz'][1]/2-self.rir_character_dict['mic']['mic_from_wall']
            mic_loc_y_range=[-mic_loc_y_range, mic_loc_y_range]
            mic_loc_y=random.uniform(*mic_loc_y_range) + self.params['room_sz'][1]/2

            
            mic_loc[:, 0]+=mic_loc_x
            mic_loc[:, 1]+=mic_loc_y

            self.params['pos_rcv']=mic_loc
            self.params['orV_rcv']=orV_rcv

            mic_center=np.array([mic_loc_x, mic_loc_y, mic_height])
            

            # check whether mic is in the room
            if 0 < mic_loc_x<self.params['room_sz'][0] and 0 < mic_loc_y<self.params['room_sz'][1] and 0 < mic_height<self.params['room_sz'][2]:
                break
        

        return theta, mic_pos, mic_center

    def get_source_pos(self, theta, azi_pos, linear_azi_pos, mic_center, room_sz):

        while True:
            r=random.uniform(*self.rir_character_dict['room']['distance']) # distance from mic
            np_azi=np.array(azi_pos)
            np_linear_azi=np.array(linear_azi_pos)

            while True:

                
                azi_deg=random.randrange(*self.rir_character_dict['room']['azimuth'])
            
               
                if len(azi_pos)==0:
                    break

         
            

                np_azi_gap=np.abs(np_azi-azi_deg)
                np_azi_360_gap=360-np_azi_gap
                np_azi_gap=np.stack((np_azi_gap, np_azi_360_gap), axis=0).min(axis=0)
               

                if azi_deg>180:
                    azi_linear_deg=360-azi_deg
                else:
                    azi_linear_deg=azi_deg

                np_linear_azi_gap=np.abs(np_linear_azi-azi_linear_deg)

                
  

                
                np_azi_gap=np_azi_gap>self.rir_character_dict['azi_gap']
                np_linear_azi_gap=np_linear_azi_gap>self.rir_character_dict['azi_gap']
            
                if np_azi_gap.all() and np_linear_azi_gap.all():
              
                    
                    break  
     

            azi_fluctuation=0.0
            

            azi=np.deg2rad(azi_deg+theta+azi_fluctuation+self.rir_character_dict['ref_vec'])
            


            ele=random.uniform(*self.rir_character_dict['room']['elevation'][:2])
            
            ele=np.deg2rad(ele)



            x=r*np.sin(ele)*np.cos(azi)
            y=r*np.sin(ele)*np.sin(azi)
            z=r*np.cos(ele)


           
           
            speech_pos=mic_center+ np.array([x,y,z])

           
            
            if 0<speech_pos[0]<room_sz[0] and 0<speech_pos[1]<room_sz[1] and 0<speech_pos[2]<room_sz[2]:
              
                break

        return speech_pos, azi_deg
    
    def get_noise_source_pos(self, theta, azi_pos, mic_center, room_sz):
        while True:
            r=random.uniform(*self.rir_character_dict['room']['distance']) # distance from mic
            np_azi=np.array(azi_pos)

            while True:


                azi_deg=random.uniform(0, 360)
        

                if len(azi_pos)==0:
                    break

                np_azi=np.abs(np_azi-azi_deg)
                np_azi_360=360-np_azi
                np_azi=np.stack((np_azi, np_azi_360), axis=0).min(axis=0)
               
                np_azi=np_azi>self.rir_character_dict['azi_gap']
          
                
                if np_azi.all():

                    
                    break  


          
            azi=np.deg2rad(azi_deg+theta+self.rir_character_dict['ref_vec'])
            ele=random.uniform(0, 180)
            ele=np.deg2rad(ele)



            

            x=r*np.sin(ele)*np.cos(azi)
            y=r*np.sin(ele)*np.sin(azi)
            z=r*np.cos(ele)
            

           
           
            speech_pos=mic_center+ np.array([x,y,z])

            
            
            if 0<speech_pos[0]<room_sz[0] and 0<speech_pos[1]<room_sz[1] and 0<speech_pos[2]<room_sz[2]:
              
                break

    

        return speech_pos, azi_deg

    def create_param(self, num_spk, with_coherent_noise, mic_type, mic_num):
        room_sz, rt60, abs_weight=self.random_room_select()    
        mic=self.whole_mic_setup
        mic_orV=mic['mic_orV']

        if mic_type=='whole':
            mic_pos=mic['mic_pos']
            mic_orV=mic['mic_orV']
            n_mic=mic_pos.shape[0]
            
        elif mic_type=='circular':
            if mic_num==4:
                mic_pos=mic['mic_pos'][:4]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][4:10]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][10:18]

        elif mic_type=='ellipsoid':
            if mic_num==4:
                mic_pos=mic['mic_pos'][18:22]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][22:28]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][28:36]

        elif mic_type=='linear':
            if mic_num==4:
                mic_pos=mic['mic_pos'][38:42]
                # mic_orV=mic['mic_orV'][:4]
            elif mic_num==6:
                mic_pos=mic['mic_pos'][37:43]
                # mic_orV=mic['mic_orV'][4:10]
            elif mic_num==8:
                mic_pos=mic['mic_pos'][36:]
        n_mic=mic_pos.shape[0]
        

       
        self.gpu_rir_param(room_sz, rt60, abs_weight)

        theta, mic_pos, mic_center= self.mic_rotate_location(mic_pos, n_mic, room_sz, mic_orV)
      
        azi_pos=[]
        linear_azi_pos=[]
        speech_pos_list=[]
        

        for i in range(num_spk):
            speech_pos, azi=self.get_source_pos(theta, azi_pos, linear_azi_pos, mic_center, room_sz)
            speech_pos_list.append(speech_pos)
            azi_pos.append(azi)
            if azi>180:
                gp=azi-180
                linear_azi_pos.append(180-gp)
            else:
                linear_azi_pos.append(azi)
           
        
        if with_coherent_noise:
            
            noise_pos, azi=self.get_noise_source_pos(theta, azi_pos, mic_center, room_sz)
        
            speech_pos_list.append(noise_pos)
            azi_pos.append(azi)
     
        self.params['pos_src']=np.stack(speech_pos_list, axis=0)
        
      
        return self.params, mic_pos, azi_pos, linear_azi_pos
    
    def room_choice_from_dict(self, room_dict):
        room_type=random.choice(list(room_dict.keys()))
        room_type=room_dict[room_type]
        room_sz=room_type['L']
        rt60=room_type['reverberation_time']
        att_max=room_type['att_max']
        att_diff=room_type['att_diff']
        abs_weight=0.5*np.ones(6)
        return room_sz, rt60, att_diff, att_max, abs_weight

    def create_param_test(self, room_dict):
        room_sz, rt60, att_diff, att_max, abs_weight = self.room_choice_from_dict(room_dict)
       

        
        self.gpu_rir_param(room_sz, rt60, abs_weight, att_diff, att_max)
       


        theta, mic_center=self.get_theta_mic_center()
     
  

        azi_pos=[]
        speech_pos_list=[]

        

        for i in range(2):
            speech_pos, azi=self.get_source_pos(theta, azi_pos, mic_center, room_sz)
            speech_pos_list.append(speech_pos)
            azi_pos.append(azi)

        self.params['pos_src']=np.stack(speech_pos_list, axis=0)
        
        return self.params, theta, mic_center

    def create_rir(self,num_spk=1, with_coherent_noise=True, mic_type='whole', mic_num=8): 
        
        
        self.params, mic_pos, azi_list, linear_azi_pos_list=self.create_param(num_spk, with_coherent_noise, mic_type, mic_num)
        
        rirs = gpuRIR.simulateRIR(**self.params)    

       
   
        
        return rirs, mic_pos, azi_list, linear_azi_pos_list
     
        



    
class acoustic_simulator_for_test(simulator_common):

    def __init__ (self, ):
        super(acoustic_simulator_for_test, self).__init__()
             

    def create_rir(self, input_dict): 
        
       

        rirs = gpuRIR.simulateRIR(**input_dict)
        mic_pos=input_dict['pos_rcv']
       
        
        
        return rirs[0], rirs[1], mic_pos[:,:2]