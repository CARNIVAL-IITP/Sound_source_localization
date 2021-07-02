import numpy as np
import os
import rir_generator as rir
import plotly.graph_objects as go
from pathlib import Path
import parmap
from glob import glob
import librosa
from soundfile import write as audiowrite
from audiolib import audioread
import scipy.signal as ss
from tqdm import tqdm
import shutil
import csv
import time
import multiprocessing as mp

class preproc_dataset():

    def __init__(self):
        print('hi')

    def split_sound_tr_cv_tt(self):
        """
        split speech -> training / validation (9:1)
        :return: save 'source_tr_cv_tt_list.npy'
        """

        tr_spc_path = '/home/dail/PycharmProjects/DCCRN/data/tr/clean/*.wav'
        tt_spc_path = os.getcwd() + '/Data/tt/clean/*.wav'
        # list_save_path = os.getcwd()

        split_dict = dict()
        tr_cv_tt = dict()

        s_list = glob(tr_spc_path)
        s_list.sort()

        # split tr:cv = 9:1
        ntr = int(np.ceil(len(s_list)*0.9))
        # ncv_tt = len(s_list) - ntr
        tr_list = s_list[:ntr]
        cv_list = s_list[ntr:]

        tt_list = glob(tt_spc_path)
        tt_list.sort()

        print('number of sound list=> tr: {}, cv: {}, tt: {}'.format(len(tr_list), len(cv_list), len(tt_list)))
        tr_cv_tt['tr'] = tr_list
        tr_cv_tt['cv'] = cv_list
        tr_cv_tt['tt'] = tt_list
        print('split tr_tt_cv list done')

        np.save(os.getcwd() + '/source_tr_cv_tt_list', tr_cv_tt)

    def copy(self):
        """
        (1) load 'source_tr_cv_tt_list.npy'
        (2) split and save tr/cv/tt speech files
        :return:
        """
        # read
        list_path = os.getcwd() + '/source_tr_cv_tt_list.npy'
        tr_cv_tt_list = np.load(list_path, allow_pickle=True).item()

        for key, value in tr_cv_tt_list.items():
            print(key)
            print(value)

            if key == 'tr':
                for i, file in enumerate(value):
                    shutil.copyfile(file, os.getcwd() + '/Data/tr/clean/' + os.path.basename(file))

            if key == 'cv':
                for i, file in enumerate(value):
                    shutil.copyfile(file, os.getcwd() + '/Data/cv/clean/' + os.path.basename(file))


    def mix_rir_and_sound_source(self, mode):
        """
        convolve speech and speech_rir (random selected)
        :param mode: tr/cv/tt
        :return: save multi-channel speech
        """
        # path set
        save_path = os.getcwd() + '/multi_channel_speech/' + mode
        rir_path = os.getcwd() + '/rir/' + mode
        if mode == 'cv':
            rir_path = os.getcwd() + '/rir/tr'
        spc_path = '/home/dail/PycharmProjects/DCCRN/data/tr/clean'

        # rir list and sound source list
        rir_list = glob(rir_path + '/*/*.npz')
        spc_list = glob(spc_path + '/*.wav')

        # generate random rir index
        spc_list.sort()
        _use_par = False

        if _use_par == True:
            if mode == 'tr':
                _ = parmap.map(self.convolve_and_save_rir_tr, spc_list, pm_pbar=True, pm_processes=28)
            if mode == 'cv':
                _ = parmap.map(self.convolve_and_save_rir_cv, spc_list, pm_pbar=True, pm_processes=28)
            if mode == 'tt':
                _ = parmap.map(self.convolve_and_save_rir_tt, spc_list, pm_pbar=True, pm_processes=28)

        else:
            for i, _spc in enumerate(tqdm(spc_list)):

                # read audio file
                # aud, fs = librosa.core.load(_spc, sr=None, mono=False)
                aud, fs = audioread(_spc)

                if len(aud.shape) != 1:
                    aud = aud[:, 0]

                #aud.shape[1]
                idx_s = np.random.randint(0, len(rir_list))
                npz = np.load(rir_list[idx_s], allow_pickle=True)

                # convolve
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path + '/' + rir_list[idx_s].split('/')[-2] + '_' + rir_list[idx_s].split('/')[-1].split('.n')[0]+'_'+_spc.split('/')[-1], Y, fs)
                # audiowrite(
                #     save_path + '/' + rir_list[idx_s].split('/')[-2] + '_' + rir_list[idx_s].split('/')[-1].split('.n')[0] + '_' + fn.split('/')[-1], Y, fs)

    def convolve_and_save_rir_tr(self, fn):

        # path set
        mode = 'tr'
        save_path = os.getcwd() + '/multi_channel_speech/' + mode + '/clean'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        rir_path = os.getcwd() + '/rir/' + mode

        mix_all = False
        rir_list = glob(rir_path + '/*/*.npz')
        # aud, fs = librosa.core.load(fn, sr=None, mono=False)
        aud, fs = audioread(fn)

        if len(aud.shape) != 1:
            aud = aud[:, 0]

        if mix_all == True:
            for i, _rir in enumerate(rir_list):
                npz = np.load(_rir, allow_pickle=True)
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path +'/'+_rir.split('/')[-2] +'_' + _rir.split('/')[-1].split('.')[0] + '_' + fn.split('/')[-1], Y, fs)
        else:
            idx_s = np.random.randint(0, len(rir_list))
            npz = np.load(rir_list[idx_s], allow_pickle=True)

            # convolve
            rir = npz['rir']
            Y = ss.convolve(rir, aud[:, np.newaxis])
            audiowrite(save_path +'/'+ rir_list[idx_s].split('/')[-2] + '_' + rir_list[idx_s].split('/')[-1].split('.n')[0] + '_' + fn.split('/')[-1], Y, fs)

    def convolve_and_save_rir_cv(self, fn):

        # path set
        mode = 'cv'
        save_path = os.getcwd() + '/multi_channel_speech/' + mode + '/clean'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        rir_path = os.getcwd() + '/rir/tr'

        mix_all = False
        rir_list = glob(rir_path + '/*/*.npz')
        # aud, fs = librosa.core.load(fn, sr=None, mono=False)
        aud, fs = audioread(fn)

        if len(aud.shape) != 1:
            aud = aud[:, 0]

        if mix_all == True:
            for i, _rir in enumerate(rir_list):
                npz = np.load(_rir, allow_pickle=True)
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path + '/' + _rir.split('/')[-1].split('.')[0] + '_' + fn.split('/')[-1], Y, fs)
        else:
            idx_s = np.random.randint(0, len(rir_list))
            npz = np.load(rir_list[idx_s], allow_pickle=True)

            # convolve
            rir = npz['rir']
            Y = ss.convolve(rir, aud[:, np.newaxis])
            audiowrite(save_path+'/'+ rir_list[idx_s].split('/')[-2]  + '_' + rir_list[idx_s].split('/')[-1].split('.n')[0] + '_' + fn.split('/')[-1], Y, fs)

    def convolve_and_save_rir_tt(self, fn):

        # path set
        mode = 'tt'
        save_path = os.getcwd() + '/multi_channel_speech/' + mode + '/clean'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        rir_path = os.getcwd() + '/rir/' + mode

        mix_all = False
        rir_list = glob(rir_path + '/*/*.npz')
        # aud, fs = librosa.core.load(fn, sr=None, mono=False)
        aud, fs = audioread(fn)

        if len(aud.shape) != 1:
            aud = aud[:, 0]

        if mix_all == True:
            for i, _rir in enumerate(rir_list):
                npz = np.load(_rir, allow_pickle=True)
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path + '/' + _rir.split('/')[-1].split('.')[0] + '_' + fn.split('/')[-1], Y, fs)
        else:
            idx_s = np.random.randint(0, len(rir_list))
            npz = np.load(rir_list[idx_s], allow_pickle=True)

            # convolve
            rir = npz['rir']
            Y = ss.convolve(rir, aud[:, np.newaxis])
            audiowrite(save_path+'/'+ rir_list[idx_s].split('/')[-2]  + '_' + rir_list[idx_s].split('/')[-1].split('.n')[0] + '_' + fn.split('/')[-1], Y, fs)


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
            ele_list = ele_list[ele_list <= 90]
            #ele_list = np.linspace(0, 90, resolution[1])
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


    def create_rir(self):
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


        params = self.get_config_for_819()
        save_path = '../Data/rir/819/'
        self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        params = self.get_config_for_409()
        save_path = '../Data/rir/409/'
        self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        params = self.get_config_for_by3()
        save_path = '../Data/rir/by3/'
        self.generate_rir_for_room(params, save_path, n_mic, r, azi, mic_pos, mic_type)

        

    def generate_rir_for_room(self, params, save_path, n_mic_list, r_list, azi_list,mic_pos_list, mic_type_list, mode='tr'):
        # Create RIR for train (R1)
        #save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R1/'
        _use_par = True

        Path(save_path).mkdir(parents=True, exist_ok=True)
        center=[params['L'][0]*0.5, params['L'][1]*0.5, 0.69]        
        
        # params['r'] = mic_pos
        self.params = params
        self.save_path = save_path

        src_pos_list = []
        for r in r_list:
            src_pos_list += self.get_uniform_dist_circ_pos(center=center, radius=r, resolution=30, dim='2D')

        
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
                pool = mp.Pool(8)
                for source_pos in src_pos_list:
                    # self.params['s']=source_pos
                    pool.apply_async(
                        self.generate_rir,
                        args=(source_pos, center, n_mic, mic_type)
                        )               
        
                pool.close()
                pool.join()
       
        # if _use_par == True:
        #     _ = parmap.map(self.generate_rir, src_pos_list)

class clean_source_mixer():

    def __init__(self):
        self.fs = 16000

    def snr_mix(self, clean, noise, target_snr, eps=1e-8):
        """
        :param clean: 4ch raw waveform
        :param noise: 4ch raw waveform
        :param snr: -5, 0, 5, 10, 15, 20 [dB]
        :param eps: 1e-8
        :return: noisy, clean, noise raw waveform
        """

        # clean_rms = self._calculate_rms(clean)
        clean_rms = (clean[0, :]**2).mean()**0.5
        clean_rms = np.maximum(clean_rms, eps)

        # noise_rms = self._calculate_rms(noise)
        noise_rms = (noise[0, :]**2).mean()**0.5
        noise_rms = np.maximum(noise_rms, eps)

        # mix speech and noise with SNR
        k = clean_rms / (10 ** (target_snr / 20)) / noise_rms

        new_noise = noise * k
        noisy = clean + new_noise

        return noisy, clean, new_noise

    def _calculate_rms(self, signal):
        """
        calc rms of wav
        """
        energy = signal ** 2
        max_e = np.max(energy)
        low_thres = max_e * (10 ** (-50 / 10))  # to filter lower than 50dB
        RMS = np.mean(energy[energy >= low_thres])

        return RMS

    def mix_spc_noi_0401(self, mode):
        """
        (1) load multi-channel speech
        (2) calculate noise rir (=speech rir + 90/180/270 degree)
        (3) convolve noise (random selected) and noise rir
        (4) mix multi-channel speech and multi-channel noise with SNR (random selected)
        :param mode: tr/cv/tt
        :return: save noisy(mix), clean(s1), noise(s2) files / save 'output.csv' file
        """

        # path set
        spc_path = os.getcwd() + '/multi_channel_speech/' + mode
        noi_path = os.getcwd() + '/Data/' + mode + '/noise'
        snr_list = [-5, 0, 5, 10, 15, 20]
        save_path = os.getcwd() + '/output/' + mode
        Path(save_path + '/mix').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/s1').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/s2').mkdir(parents=True, exist_ok=True)

        # multi-channel speech list
        s_list = glob(spc_path + '/*.wav')
        # single-channel noise list
        n_list = glob(noi_path + '/*.wav')

        # make 'output.csv'
        f = open(f'output/{mode}/output.csv', 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['order', 'speech', 'room', 'speech_rir', 'noise', 'noise_rir', 'azimuth', 'snr'])

        for i, s in enumerate(s_list):
            multi_ch_aud, fs = librosa.core.load(s, sr=None, mono=False)
            multi_ch_aud_na = os.path.splitext(os.path.basename(s))[0]

            # select noise azimuth
            split = multi_ch_aud_na.split('_')
            spc_na = f'{split[-2]}_{split[-1]}'
            spc_rir_na =f'{split[1]}_{split[2]}_{split[3]}'
            # print(time.time() - start)

            # noise rir = speech rir + 90/180/270 degree
            az = int(split[1][2:])
            room = split[0]
            n = np.random.randint(1, 4)
            noi_az = (az + 90*n) % 360  # +90 +180 +270 degree
            split = multi_ch_aud_na.split('_')
            noi_rir_na = f'az{noi_az}_{split[2]}_{split[3]}'
            noi_rir = os.getcwd() + f'/rir/{mode}/{split[0]}/{noi_rir_na}.npz'
            if mode == 'cv':
                noi_rir = os.getcwd() + f'/rir/tr/{split[0]}/{noi_rir_na}.npz'

            # select and load random noise
            idx_n = np.random.randint(0, len(n_list))
            noi, fs2 = librosa.core.load(n_list[idx_n], sr=None)
            noi_na = os.path.splitext(os.path.basename(n_list[idx_n]))[0]
            assert fs == fs2

            # convolve noise with RIR
            npz = np.load(noi_rir, allow_pickle=True)
            rir = npz['rir']
            rand_start = np.random.randint(0, noi.shape[0] - multi_ch_aud.shape[1] - 8191)
            multi_ch_noi_tmp = ss.convolve(rir, noi[rand_start:rand_start + multi_ch_aud.shape[1] + 8191, np.newaxis])

            multi_ch_noi = multi_ch_noi_tmp[8191:-8191, :].transpose()


            # mix speech and noise with SNR
            idx_snr = np.random.randint(0, len(snr_list))
            snr = snr_list[idx_snr]
            noisy, clean, noise = self.snr_mix(multi_ch_aud, multi_ch_noi, snr)

            audiowrite(save_path + f'/mix/noisy_{i + 1:#05d}_{noi_na}_{snr}.wav', noisy.transpose(), fs)
            audiowrite(save_path + f'/s1/clean_{i + 1:#05d}.wav', clean.transpose(), fs)
            audiowrite(save_path + f'/s2/noise_{i + 1:#05d}.wav', noise.transpose(), fs)

            wr.writerow([i, spc_na, room, spc_rir_na, noi_na, noi_rir_na, 90*n, snr])

        f.close()

    def mix_spc_noi_tt(self):
        """
        (1) load single channel speech / single channel speech
        (2) select speech rir and noise rir
        (3) convolve speech(noise) and speech(noise) rir
        (4) mix multi-channel speech and multi-channel noise
        (5) room(2) * noise(4) * SNR(5)
        :return: save noisy(mix), clean(s1), noise(s2) files / save 'output.csv' file
        """

        # path set
        spc_path = '/home/dail/PycharmProjects/DCCRN/datasets/tr/clean'
        noi_path = '/home/dail/PycharmProjects/DCCRN/datasets/tr/noise'
        snr_list = [-5, 0, 5, 10, 15]
        save_path = os.getcwd() + '/output/tt'
        Path(save_path + '/mix').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/s1').mkdir(parents=True, exist_ok=True)
        Path(save_path + '/s2').mkdir(parents=True, exist_ok=True)

        # multi-channel speech list
        s_list = glob(spc_path + '/*.wav')
        # single-channel noise list
        n_list = glob(noi_path + '/*.wav')

        # make 'output.csv'
        f = open(f'output/tt/output.csv', 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['order', 'speech', 'room', 'speech_rir', 'noise', 'noise_rir', 'snr'])
        cnt = 0

        for i, s in enumerate(s_list):
            # multi_ch_aud, fs = librosa.core.load(s, sr=None, mono=False)
            # multi_ch_aud_na = os.path.splitext(os.path.basename(s))[0]
            spc, fs = audioread(s)
            spc_na = os.path.splitext(os.path.basename(s))[0]

            # select speech/noise rir
            # np.random.seed(1)
            rand_azi_s = np.random.choice(np.concatenate((np.arange(31), np.arange(330, 360)), axis=0))
            # np.random.seed(1)
            rand_azi_n = np.random.choice(np.arange(180, 271))
            rand_r = np.round(np.random.choice(np.linspace(1, 2.2, 5)), 1)
            spc_rir_na = f'az{rand_azi_s}_el0_r{rand_r}'
            noi_rir_na = f'az{rand_azi_n}_el0_r{rand_r}'

            room = ['R4', 'R5']

            # room
            for n in range(2):
                spc_rir = os.getcwd() + f'/rir/tt/{room[n]}/{spc_rir_na}.npz'
                npz_s = np.load(spc_rir, allow_pickle=True)
                rir_s = npz_s['rir']
                multi_ch_spc = ss.convolve(rir_s, spc[:, np.newaxis])
                multi_ch_spc = multi_ch_spc.transpose()

                noi_rir = os.getcwd() + f'/rir/tt/{room[n]}/{noi_rir_na}.npz'
                npz_n = np.load(noi_rir, allow_pickle=True)
                rir_n = npz_n['rir']

                # noise
                for idx_n in range(len(n_list)):

                    noi, fs2 = librosa.core.load(n_list[idx_n], sr=None)
                    noi_na = os.path.splitext(os.path.basename(n_list[idx_n]))[0]
                    assert fs == fs2

                    rand_start = np.random.randint(0, noi.shape[0] - multi_ch_spc.shape[1] - 8191)
                    multi_ch_noi_tmp = ss.convolve(rir_n, noi[rand_start:rand_start + multi_ch_spc.shape[1] + 8191, np.newaxis])
                    multi_ch_noi = multi_ch_noi_tmp[8191:-8191, :].transpose()

                    # mix speech and noise with SNR
                    # idx_snr = np.random.randint(0, len(snr_list))

                    for l in range(len(snr_list)):
                        cnt = cnt+1
                        snr = snr_list[l]

                        noisy, clean, noise = self.snr_mix(multi_ch_spc, multi_ch_noi, snr)

                        audiowrite(save_path + f'/mix/noisy_{cnt:#05d}_{noi_na}_{snr}.wav', noisy.transpose(), fs)
                        audiowrite(save_path + f'/s1/clean_{cnt:#05d}.wav', clean.transpose(), fs)
                        audiowrite(save_path + f'/s2/noise_{cnt:#05d}.wav', noise.transpose(), fs)

                        wr.writerow([cnt, spc_na, room[n], spc_rir_na, noi_na, noi_rir_na, snr])

        f.close()

if __name__ == "__main__":

    AS = acoustic_simulator()
    PD = preproc_dataset()
    SM = clean_source_mixer()
    
    "[1] make and save RIR (R1, R2, R3, R4, R5)"
    AS.create_rir()
    # AS.generate_rir_for_room()
    exit()


    "[2] split tr/cv/tt"
    # PD.split_sound_tr_cv_tt()
    # PD.copy()

    order = ['tr', 'cv', 'tt']

    "[3] mix RIR and sound source -> multi-channel speech"
    for select in order:
        PD.mix_rir_and_sound_source(mode='tt')

    PD.mix_rir_and_sound_source(mode='tr')
    PD.mix_rir_and_sound_source(mode='cv')
    exit()

    "[4] mix RIR and noise source -> multi-channel noise / mix speech and noise with SNR"
    # for select in order:
        # SM.mix_spc_noi_0401(mode=select)
    # SM.mix_spc_noi_0401(mode='tr')
    # SM.mix_spc_noi_0401(mode='cv')
    # SM.mix_spc_noi_tt()
