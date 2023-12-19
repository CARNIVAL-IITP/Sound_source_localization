import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util import *
import pandas as pd
import sklearn
import random
import math
import copy
import pickle
import yaml
import torch

def randomseed_init(num):
    np.random.seed(num)
    random.seed(num)
    torch.manual_seed(num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num)
        return 'cuda'
    else:
        return 'cpu'

def load_yaml(yaml_dir):
    yaml_file=open(yaml_dir, 'r')
    data=yaml.safe_load(yaml_file)
    yaml_file.close()
    return data

class val_csv_prepare():
    def __init__(self, config, config_type):
        
        self.config=load_yaml(config)


   

        _=randomseed_init(self.config['randomseed'])
        self.max_spk=self.config['max_spk']
        self.duration = self.config['duration']
        self.speech_least_chunk_size=self.config['speech_least_chunk_size']

        self.total_csv=self.config['save_csv']        
        self.save_dict=self.config['save_dict']
        self.each_spk_dict=self.config['each_spk_dict']

        self.config=self.config[config_type]

        self.mic_character_dict=self.config['mic']
        self.sound_pos_dict=self.config['sound_pos']
        self.train_len=self.config['train_len']
        

        self.data_dict_save_dir=self.config['data_dict_dir']
        os.makedirs(self.data_dict_save_dir, exist_ok=True)
 
        

        data_csv_dir=self.config['data_csv_dir']
        self.noise_csv=pd.read_csv(data_csv_dir+self.config['noise_csv'], index_col=0)       
        self.speech_csv=pd.read_csv(data_csv_dir+self.config['speech_csv'], index_col=0)
        
        self.save_csv_dir=data_csv_dir+self.config['save_csv']


        circle=self.circle_mic_pos()
        ellipsoid=self.ellip_mic_pos()
        linear=self.linear_mic_pos()
        self.mic_pos_dict={'circle':circle, 'ellipsoid':ellipsoid, 'linear':linear}

        self.run()
    
    def random_room_select(self):
      
        room_character=self.config['room']

        
        room_sz_bound=room_character['room_sz_bound']
        x=random.uniform(*room_sz_bound['x'])
        y=random.uniform(*room_sz_bound['y'])
        z=random.uniform(*room_sz_bound['z'])
        

        rt60=random.uniform(*room_character['rt60_bound'])
        


        att_diff=random.uniform(*room_character['att_diff'])
        att_max=random.uniform(*room_character['att_max'])

        abs_weight=np.random.uniform(*room_character['abs_weights_bound'], size=6)
        

        return [x,y,z], rt60, att_diff, att_max, abs_weight
        
    def run(self):
        speech_csv=sklearn.utils.shuffle(self.speech_csv)
        noise_csv=sklearn.utils.shuffle(self.noise_csv)

        for iter_num, speech_row_data in enumerate(speech_csv.iterrows()):

            if iter_num%100==0:
                print(iter_num, len(speech_csv))

            current_save_dict=copy.deepcopy(self.save_dict)
            azi_list=[]


            ###### 공통 사항
            
            noramlize_factor=random.uniform(*self.config['normalize_factor'])
            current_save_dict['normalize_factor']=noramlize_factor


            ############## room_shape

            room_sz, rt60, att_diff, att_max, abs_weight=self.random_room_select()

            
        pd.DataFrame(self.total_csv).to_csv(self.save_csv_dir)
        
        
  

    def circle_mic_pos(self):
        pos_4=np.array([[2.828427125, 0],
                        [0, 2.828427125],
                        [-2.828427125, 0],
                        [0, -2.828427125]])

        pos_6=np.array([[4, 0],
                        [2, 3.4641016151377553],
                        [-2, 3.4641016151377557],
                        [-4,0],
                        [-2, -3.464101615137755],
                        [2, -3.4641016151377553]])

        pos_8=np.array([[5.226251859505506, 0],
                        [3.6955181300451474, 3.695518130045147],
                        [0, 5.226251859505506],
                        [-3.695518130045147, 3.6955181300451474],
                        [-5.226251859505506, 0],
                        [-3.695518130045148, -3.695518130045147],
                        [0,-5.226251859505506],
                        [3.695518130045146, -3.695518130045148]])
        return {4:pos_4, 6:pos_6, 8:pos_8}

    def ellip_mic_pos(self):
        pos_4=np.array([[2.4210, 0],
                        [0, 3.1841],
                        [-2.4210, 0],
                        [0, -3.1841]])

        pos_6=np.array([[4.6149, 0],
                [2, 3.0269],
                [-2, 3.2069],
                [-4.6149, 0],
                [-2, -3.2069],
                [2, -3.0269]])

    

        pos_8=np.array([[5.9229, 0],
                [3.8509, 3.4216],
                [0, 4.5033],
                [-3.8509, 3.4216],
                [-5.9229, 0],
                [-3.8509, -3.4216],
                [0, -4.5033],
                [3.8509, -3.4216]])





        return {4:pos_4, 6:pos_6, 8:pos_8}
    
    def linear_mic_pos(self):

        pos_4=np.array([[0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6]])
        pos_4=np.flip(pos_4, -1)
        
        pos_6=np.array([[0,10],
                        [0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6],
                        [0,-10]])
        pos_6=np.flip(pos_6, -1)

        pos_8=np.array([[0, 14],
                        [0,10],
                        [0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6],
                        [0,-10],
                        [0, -14]])

        pos_8=np.flip(pos_8, -1)
        return {4:pos_4, 6:pos_6, 8:pos_8}

class eval_csv_prepare():
    def __init__(self, config, config_type):
        self.config=load_yaml(config)

        _=randomseed_init(self.config['randomseed'])
        self.max_spk=self.config['max_spk']
        self.duration = self.config['duration']
        self.speech_least_chunk_size=self.config['speech_least_chunk_size']

        self.total_csv=self.config['save_csv']        
        self.save_dict=self.config['save_dict']
        self.each_spk_dict=self.config['each_spk_dict']

        self.config=self.config[config_type]
        

        self.mic_character_dict=self.config['mic']
        self.sound_pos_dict=self.config['sound_pos']
        self.train_len=self.config['train_len']
        self.data_dict_save_dir=self.config['data_dict_dir']
        

        data_csv_dir=self.config['data_csv_dir']
        self.noise_csv=pd.read_csv(data_csv_dir+self.config['noise_csv'], index_col=0)       
        self.speech_csv=pd.read_csv(data_csv_dir+self.config['speech_csv'], index_col=0)
        self.rir_csv=pd.read_csv(data_csv_dir+self.config['rir_csv'], index_col=0)
        self.save_csv_dir=data_csv_dir+self.config['save_csv']

        self.room_character_dict= load_yaml(data_csv_dir+self.config['room_character_dict'])

        circle=self.circle_mic_pos()
        ellipsoid=self.ellip_mic_pos()
        linear=self.linear_mic_pos()
        self.mic_pos_dict={'circle':circle, 'ellipsoid':ellipsoid, 'linear':linear}

        self.run()

    def random_room_select(self):
      
        room_character=self.config['room']

        
        room_sz_bound=room_character['room_sz_bound']
        x=random.uniform(*room_sz_bound['x'])
        y=random.uniform(*room_sz_bound['y'])
        z=random.uniform(*room_sz_bound['z'])
        

        rt60=random.uniform(*room_character['rt60_bound'])
        


        att_diff=random.uniform(*room_character['att_diff'])
        att_max=random.uniform(*room_character['att_max'])

        abs_weight=np.random.uniform(*room_character['abs_weights_bound'], size=6)
        

        return [x,y,z], rt60, att_diff, att_max, abs_weight
        
    def run(self):
        
        speech_csv=sklearn.utils.shuffle(self.speech_csv)
        noise_csv=sklearn.utils.shuffle(self.noise_csv)
        

        for iter_num, speech_row_data in enumerate(speech_csv.iterrows()):
            if iter_num%100==0:
                print(iter_num, len(speech_csv))

            current_save_dict=copy.deepcopy(self.save_dict)
            azi_list=[]

            ###### 공통 사항
            
            noramlize_factor=random.uniform(*self.config['normalize_factor'])
            current_save_dict['normalize_factor']=noramlize_factor


            ############## room_shape

            room_type=self.rir_csv.sample(n=1, ignore_index=True)['room_type'][0]   
            room_config=self.room_character_dict[room_type]
            room_sz=room_config['L']
            current_save_dict['room_sz']=room_config['L']
            current_save_dict['T_max']=room_config['reverberation_time']
            current_save_dict['att_max']=room_config['att_max']
            current_save_dict['att_diff']=room_config['att_diff']
            current_save_dict['abs_weight']=np.array([1.0]*6)
            
          


            ########## mic
            
            n_mic=random.choice(self.mic_character_dict['mic_num']) # number of mic 
            mic_shape=random.choice(self.mic_character_dict['mic_shape']) # mic shap

            mic_pos=self.mic_pos_dict[mic_shape][n_mic]/100

            theta=random.uniform(0, 2*math.pi)
            # theta=0.0
            # theta=math.pi/2
            c, s=np.cos(theta), np.sin(theta)      
            R=np.array(((c, -s), (s,c)))
            mic_pos=R.dot(mic_pos[:,:2].T).T      
            theta=np.rad2deg(theta)   

            mic_pos=np.pad(mic_pos, ((0,0),(0,1)))
            

            current_save_dict['mic']['mic_pos']=mic_pos
            current_save_dict['mic']['theta']=theta

            mic_height=random.uniform(*self.mic_character_dict['mic_height'])
            

            mic_loc_x_range=room_sz[0]/2-self.mic_character_dict['mic_from_wall']
            mic_loc_x_range=[-mic_loc_x_range, mic_loc_x_range]
            mic_loc_x=random.uniform(*mic_loc_x_range) + room_sz[0]/2

            mic_loc_y_range=room_sz[1]/2-self.mic_character_dict['mic_from_wall']
            mic_loc_y_range=[-mic_loc_y_range, mic_loc_y_range]
            mic_loc_y=random.uniform(*mic_loc_y_range) + room_sz[1]/2
            mic_center_loc=[mic_loc_x, mic_loc_y, mic_height]

            current_save_dict['mic']['mic_center_loc']=mic_center_loc

            ##### speech
            num_spk=random.randint(1, self.max_spk)
            current_save_dict['num_spk']=num_spk
            speech_info=self.speech_csv.iloc[iter_num:iter_num+1]
            temp_total_df=self.speech_csv
            for spk in range(num_spk-1):
                last_speaker=speech_info.iloc[-1]['speaker']
                temp_total_df=temp_total_df.drop(temp_total_df[temp_total_df['speaker']==last_speaker].index)
                temp_df=temp_total_df.sample(1)            
                speech_info=pd.concat((speech_info, temp_df))

            


            speech_azi_list = [0 for i in range(self.max_spk)]

            for spk_num, spk_info in enumerate(speech_info.iterrows()):

                speaker_dict=copy.deepcopy(self.each_spk_dict)
                spk_info=spk_info[1]
                speaker_dict['speech_wav']=spk_info['audio_directory']
                if spk_num==0:
                    speaker_dict['ref_snr']=0.0
                else:
                    speaker_dict['ref_snr']=random.uniform(*self.config['speech_snr'])


                while True:
                    r=random.uniform(*self.sound_pos_dict['distance']) # distance from mic

                    while True:
                        azi=random.randrange(*self.sound_pos_dict['azimuth'])
                        
                        if len(speech_azi_list)==0:
                            speech_azi_list[spk]=azi
                            break
                        else:
                            okay_to_append=True
                            for past_azi in speech_azi_list[:spk]:
                                gap=np.abs(past_azi-azi)
                                if gap<30 or gap>330:
                                    okay_to_append=False
                                    break

                            if okay_to_append:
                                speech_azi_list[spk]=azi
                                break
                        if azi not in azi_list:
                            azi_list.append(azi)
                            break
                    
                    speaker_dict['azi']=azi
                    
                    azi+=theta
                    azi=np.deg2rad(azi)
                    ele=90-random.randrange(*self.sound_pos_dict['elevation'])
                    ele=np.deg2rad(ele)

                    x=r*np.sin(ele)*np.cos(azi)
                    y=r*np.sin(ele)*np.sin(azi)
                    z=r*np.cos(ele)

                 
                    speech_pos=mic_center_loc+ np.array([x,y,z])

                    if 0<speech_pos[0]<room_sz[0] and 0<speech_pos[1]<room_sz[1] and 0<speech_pos[2]<room_sz[2]:
                 
                        break
            
                speaker_dict['speech_loc']=speech_pos
                speech_total_duration=spk_info['length']

                if speech_total_duration>self.duration:
                    speech_pos=random.choice(['front', 'mid', 'back'])

                    if speech_pos == 'mid':
                        speech_duration=self.duration
                        speech_start_sample=np.random.randint(0, speech_total_duration-speech_duration)
                        

                    elif speech_pos == 'back':
                        speech_duration=random.randint(self.speech_least_chunk_size, self.duration)
                        speech_start_sample=speech_total_duration-speech_duration
                   

                    elif speech_pos == 'front':
                        speech_duration=random.randint(self.speech_least_chunk_size, self.duration)
                        speech_start_sample=0
                        

                else:
                    speech_pos=random.choice(['front', 'back'])

                    if speech_pos == 'back':
                        speech_duration=random.randint(self.speech_least_chunk_size, self.duration)
                        speech_start_sample=speech_total_duration-speech_duration
                   

                    elif speech_pos == 'front':
                        speech_duration=random.randint(self.speech_least_chunk_size, self.duration)
                        speech_start_sample=0

                

             
                speaker_dict['speech_pos_type']=speech_pos
                speaker_dict['speech_start_point']=speech_start_sample
                speaker_dict['speech_duration']=speech_duration
                speaker_dict['speech_pos_type']=speech_pos
           
                current_save_dict['spk_list'].append(speaker_dict)
    

            ######### noise

            noise_row_data=noise_csv.iloc[iter_num%len(noise_csv)]
            noise_len=noise_row_data['length']    
            current_save_dict['noise']['noise_wav']=noise_row_data['noise_directory']

          
            noise_start_point=random.randrange(0, noise_len-self.train_len)
           
            
            current_save_dict['noise']['noise_start_point']=noise_start_point

            snr=random.uniform(*self.config['SNR'])
            current_save_dict['noise']['SNR']=snr

            while True:
                r=random.uniform(*self.sound_pos_dict['distance']) # distance from mic

                while True:
                    azi=random.randrange(*self.sound_pos_dict['azimuth'])
                    if azi not in azi_list[:num_spk]:
                        azi_list.append(azi)
                        break
                current_save_dict['noise']['azi']=azi
                azi+=theta
                azi=np.deg2rad(azi)                
                ele=90-random.randrange(*self.sound_pos_dict['elevation'])
                ele=np.deg2rad(ele)

                x=r*np.sin(ele)*np.cos(azi)
                y=r*np.sin(ele)*np.sin(azi)
                z=r*np.cos(ele)

                noise_pos=mic_center_loc+ np.array([x,y,z])
                if 0<noise_pos[0]<room_sz[0] and 0<noise_pos[1]<room_sz[1] and 0<noise_pos[2]<room_sz[2]:
                    break
            
            current_save_dict['noise']['noise_loc']=noise_pos
            file_name=str(iter_num)+'.pkl'
            self.total_csv['data_list'].append(file_name)

            output=open(self.data_dict_save_dir+file_name, 'wb')
            pickle.dump(current_save_dict, output)
            output.close()
           
        pd.DataFrame(self.total_csv).to_csv(self.save_csv_dir)
        
  

    def circle_mic_pos(self):
        pos_4=np.array([[2.828427125, 0],
                        [0, 2.828427125],
                        [-2.828427125, 0],
                        [0, -2.828427125]])

        pos_6=np.array([[4, 0],
                        [2, 3.4641016151377553],
                        [-2, 3.4641016151377557],
                        [-4,0],
                        [-2, -3.464101615137755],
                        [2, -3.4641016151377553]])

        pos_8=np.array([[5.226251859505506, 0],
                        [3.6955181300451474, 3.695518130045147],
                        [0, 5.226251859505506],
                        [-3.695518130045147, 3.6955181300451474],
                        [-5.226251859505506, 0],
                        [-3.695518130045148, -3.695518130045147],
                        [0,-5.226251859505506],
                        [3.695518130045146, -3.695518130045148]])
        return {4:pos_4, 6:pos_6, 8:pos_8}

    def ellip_mic_pos(self):
        pos_4=np.array([[2.4210, 0],
                        [0, 3.1841],
                        [-2.4210, 0],
                        [0, -3.1841]])

        pos_6=np.array([[4.6149, 0],
                [2, 3.0269],
                [-2, 3.2069],
                [-4.6149, 0],
                [-2, -3.2069],
                [2, -3.0269]])

    

        pos_8=np.array([[5.9229, 0],
                [3.8509, 3.4216],
                [0, 4.5033],
                [-3.8509, 3.4216],
                [-5.9229, 0],
                [-3.8509, -3.4216],
                [0, -4.5033],
                [3.8509, -3.4216]])





        return {4:pos_4, 6:pos_6, 8:pos_8}
    
    def linear_mic_pos(self):

        pos_4=np.array([[0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6]])
        pos_4=np.flip(pos_4, -1)
        
        pos_6=np.array([[0,10],
                        [0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6],
                        [0,-10]])
        pos_6=np.flip(pos_6, -1)

        pos_8=np.array([[0, 14],
                        [0,10],
                        [0, 6],
                        [0, 2],
                        [0, -2],
                        [0, -6],
                        [0,-10],
                        [0, -14]])
        pos_8=np.flip(pos_8, -1)
        return {4:pos_4, 6:pos_6, 8:pos_8}