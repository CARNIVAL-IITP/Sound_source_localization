import numpy as np
import os
import rir_generator as rir
import plotly.graph_objects as go
from pathlib import Path
import parmap
from glob import glob
import librosa
from soundfile import write as audiowrite
# from audiolib import audioread
import scipy.signal as ss
from tqdm import tqdm
import shutil
import csv
import time
import multiprocessing as mp
import yaml


class acoustic_simulator():

    def __init__ (self):
        print('acoustic simulator v1')
        self.initialize_room_params()
        self.save_path = os.getcwd() + '/rir'
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    

    def get_config_for_819(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [7.9, 7.0, 2.7]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []] # mic location
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2] # source location
        params['reverberation_time'] = 0.2
        params['nsample'] = 16000
        return params

    def get_config_for_409(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [7.0,4.2, 2.7]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []] # mic location
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2] # source location
        params['reverberation_time'] = 0.25
        params['nsample'] = 16000
        return params

    def get_config_for_by3(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [8.3, 3.4, 2.5]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []] # mic location
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2] # source location
        params['reverberation_time'] = 0.3
        params['nsample'] = 16000
        return params

    def initialize_room_params(self):
        # rir set
        self.params = dict()
        self.params['c'] = 343 # sound velocity
        self.params['fs'] = 16000 # sampling frequency
        self.params['L'] = [8, 8, 6] # Room dimensions
        self.params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        self.params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        self.params['reverberation_time'] = 0.0
        self.params['nsample'] = 8192

        # microphone /source set
        self.radius_mic = 0.04
        self.radius_src = 2
        self.nmic = 4
        self.resolution = [10, 10] # azimuth/elevation angle resolution,

        # for visual plotting
        self.visualization = False
        self.dim = '3D'

    def get_UCA_array(self, center, radius, nmic, visualization=True):
        """
        입력: center (list)을 기준으로
        :return: 마이크 중심으로 반지름 만큼 떨어진 4개의 마이크 위치
        """
        # convert numpy array
        center = np.array(center)

        # center + [r*cosd(mic_resol(i)), r*sind(mic_resol(i)), 0];
        #mic_resol = np.linspace(0, 360, nmic + 1)[:-1]
        #mic_resol = mic_resol[:-1]

        resol = 360 / nmic
        mic_resol = np.linspace(0, 360 - resol, nmic)

        pos_mics = []
        for i in range(nmic):
            # numerical error로부터 안전해지기 위해 소숫점 3번쨰 자리에서 반올림
            pos_mics.append(center + np.round(np.array([radius*np.cos(mic_resol[i]*np.pi/180), radius*np.sin(mic_resol[i]*np.pi/180), 0]), 3))
            print("position at mic{}: {}".format(i+1, pos_mics[i]))

        if visualization == True:
            # 마이크 위치랑 방을 보여줌
            self.visualize_pos(pos_mics)
        return pos_mics

    def visualize_pos(self, rec_list):
        """
        음원 source와 receiver를 visualization
        :return:ddd
        """
        rec_list = np.array(rec_list)  # [ndata, (x,y,z)]

        # ## Interactive visualization using plotly
        fig = go.Figure(data=[go.Scatter3d(x=rec_list[:, 0],
                                           y=rec_list[:, 1],
                                           z=rec_list[:, 2],
                                           mode='markers',
                                           marker=dict(
                                               color='red',
                                               size=5
                                           )
                                           )])

        fig.show()
        

    def get_uniform_dist_circ_pos(self, center, radius, resolution, dim):
        """
        center를 기준으로 원형으로 퍼진
        # elevation angle 0~90 사이
        # azimuth angle 0~360 사이
        :param center:
        :param radius: 1[m]
        :param resolution: [azi, ele] 각도
        :return:
        """
        #resol = 360 / nresol
        pos_mics = []

        if dim == '3D':
            azi_list = np.arange(0, 360, resolution[0])
            ele_list = np.arange(0, 60+resolution[1], resolution[1])
            # print(ele_list)
            # exit()
            # ele_list = ele_list[ele_list <= 90]
            #ele_list = np.linspace(0, 90, resolution[1])
            ele_list=ele_list[:-1]
            # print(ele_list)
            # exit()
        else:
            azi_list = np.arange(0, 360, resolution)
            ele_list = [0]

        for ele in ele_list:
            for azi in azi_list:
                pos_mics.append(center + self.sph2cart(radius, azi, ele))

        return pos_mics

    def generate_rir(self, src_pos, center, n_mic, mic_type):
       
        self.params['s'] = src_pos
        # center = (np.array(self.params['L']) / 2).tolist()
        azi, ele, r = self.cart2sph(src_pos - center)
        h = rir.generate(**self.params)
        
        np.savez(self.save_path + '{}_n{}_az{}_el{}_r{}.npz'.format(mic_type, n_mic, int(azi), int(ele), r), rir=h, params=self.params)

    def cart2sph(self, position):
        x, y, z = position
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)

        # rad2deg
        az = np.round(az * 180 / np.pi, 0)
        el = np.round(el * 180 / np.pi, 0)

        if az < 0:
            az = np.abs(az) + 180

        return az, el, np.round(r, 1)

    def sph2cart(self, radius, azimuth, elevation):
        """
        :param radius: in m
        :param azimuth: in deg
        :param elevation: in deg
        :return:
        """
        rcos_theta = radius * np.cos(elevation*np.pi/180)
        x = rcos_theta * np.cos(azimuth*np.pi/180)
        y = rcos_theta * np.sin(azimuth*np.pi/180)
        z = radius * np.sin(elevation*np.pi/180)

        return np.round(np.array([x, y, z]), 3)

    def circle_mic_pos(self):
        pos_4=np.array([[2.8284271247461903, 0],
                        [0, 2.8284271247461903],
                        [-2.8284271247461903, 0],
                        [0, -2.8284271247461903]])

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


    def create_rir(self, params, save_path):
        # save_name = room type _ # of mic_ r _ azi_ array type
        _use_par = True
        n_mic=[4,6,8]
        mic_type={'circle', 'ellipsoid', 'linear'}
        r=[0.5, 1.0, 1.5]
        azi=[30*i for i in range(12)]
        circle=self.circle_mic_pos()
        ellipsoid=self.ellip_mic_pos()
        linear=self.linear_mic_pos()
        mic_pos={'circle':circle, 'ellipsoid':ellipsoid, 'linear':linear}


        # params = self.get_config_for_819()
        # save_path = '../Data/rir/819/'
        # self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        # params = self.get_config_for_409()
        # save_path = '../Data/rir/409/'
        # self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        # params = self.get_config_for_by3()
        # save_path = '../Data/rir/by3/'
        self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        

    def generate_rir_for_room(self, params, save_path, n_mic_list, r_list, azi_list,mic_pos_list, mic_type_list, mode='tr'):
        # Create RIR for train (R1)
        #save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R1/'
        _use_par = True

        
        center=[params['L'][0]*0.5, params['L'][1]*0.5, 0.69]        
        
        # params['r'] = mic_pos
        self.params = params
        self.save_path = save_path

        src_pos_list = []
        
        for r in r_list:
            src_pos_list += self.get_uniform_dist_circ_pos(center=center, radius=r, resolution=[30,15], dim='3D')

        # print(len(src_pos_list))
        # exit()
        
        # mgr = mp.Manager()
        for n_mic in n_mic_list:
            for mic_type in mic_type_list:
                print(n_mic, mic_type)
                # mic_type='ellipsoid'
                # mic_type='linear'
                # n_mic=8
                # print(mic_type)
                pos=mic_pos_list[mic_type][n_mic]/100
                
                # print(pos)
                # exit()
                self.params['r']=np.pad(pos, ((0,0), (0,1)))+center
                
                # self.visualize_pos(np.pad(pos, ((0,0), (0,1))))
                # # self.visualize_pos(self.params['r'])
                # exit()
                pool = mp.Pool(6)
                for source_pos in src_pos_list:
                    # self.params['s']=source_pos
                    pool.apply_async(
                        self.generate_rir,
                        args=(source_pos, center, n_mic, mic_type), 
                        )               
        
                pool.close()
                pool.join()
       
        # if _use_par == True:
        #     _ = parmap.map(self.generate_rir, src_pos_list)



def load_yaml(yaml_file):
    yaml_file=open(yaml_file, 'r')
    data=yaml.load(yaml_file)
    f=list(data.keys())
    return data, f

def rir_param_prepare(data, loc, room_type):
    data['nsample']=int(data['fs']*data['reverberation_time'])
    loc=loc+'/'+room_type+'/'
    os.makedirs(loc, exist_ok=True)
    
    
    return data, loc

if __name__ == "__main__":
    room_info, room_name=load_yaml('./room_character.yaml')
    AS = acoustic_simulator()
    rir_save_path=AS.save_path

    for room_type in room_name:   
        param, save_path=rir_param_prepare(room_info[room_type],rir_save_path, room_type)
        AS.create_rir(param, save_path)

    # AS.generate_rir_for_room()
