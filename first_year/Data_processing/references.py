import torch
import torch.nn as nn
import numpy as np
import librosa
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import parmap
from soundfile import write as audiowrite
from librosa.core import load as audioread
from glob import glob
from cls_feat_extract import feature_extractor
from pathlib import Path
from tqdm import tqdm

class preproc_dataset():

    def __init__(self):
        print('hi')
        self.cls_feat = feature_extractor()
        self.save_path ='/media/jeonghwan/HDD2/IS2021/dataset/sound_source/sound/'
    # def preproc_clean_dataset(self):
    #     """
    #     6초 단위로 잘려져 있는 48kHz dataset을 22.05 kHz로 down sampling하여 다시 저장
    #     :return:
    #     """
    #     input_path = '/media/jeonghwan/HDD2/Dataset/clean_label/clean_cut6/'
    #     npz_list = glob(input_path + '*.npz')
    #     for i, npz in enumerate(npz_list):
    #         npz_temp = np.load(npz, allow_pickle=True)
    #         aud = npz_temp['aud']
    #
    #         # down sampling
    #         aud_re = librosa.resample(aud, 48000, 22050)
    #
    #         # sample to sec
    #         label = npz_temp['label']
    #
    #         # find begin point and end point
    #         prev_sad = 0
    #         for i, sad in enumerate(label):
    #             if (prev_sad==0) and (sad==1):
    #                 beg_pt =
    #             elif (prev_sad==1) and (sad=0):
    #                 end_pt =
    #
    #         # sec to sample

    #def sec2sample:

    #def sample2sec:
    def preproc_db(self):
        target_path = '/media/jeonghwan/HDD2/Dataset/clean_label/clean_cut/'
        save_path = '/media/jeonghwan/HDD2/Dataset/clean_label/22Khz/'
        npy_list = glob(target_path + '*.npy')
        for i, npy_ in enumerate(npy_list):
            beg_end = np.load(npy_, allow_pickle=True)
            print(beg_end)

        rec1_npy = np.load(target_path + 'rec1_beg_end.npy', allow_pickle=True)

        aud, fs = self.cls_feat.audioread(target_path + 'rec1.wav')
        aud_re = librosa.resample(aud, fs, 22050) # resampling

        text_file = open(target_path + 'rec1_label.txt', 'r')
        lines = text_file.readlines()

        for i, line in enumerate(lines):
            print(line)
            beg = float(line.split('\t')[0])
            end = float(line.split('\t')[1])

        print(rec1_npy)
    # 먼저 길
    def preproce_db_210301(self):
        # /media/jeonghwan/HDD2/Dataset/clean_label/clean_cut6

        aud_path = '/media/jeonghwan/HDD2/Dataset/clean_label/clean_cut6/'


        npz_list = glob(aud_path + '*.npz')
        plt.figure(1)
        # for i, npz_ in enumerate(tqdm(npz_list)):
        #     npz = np.load(npz_, allow_pickle=True)
        #     aud = npz['aud']
        #     label = npz['label']
        #
        #     # downsample
        #     aud_re = librosa.resample(aud, 48000, 16000) # resampling
        #
        #     # label downsample
        #     label = label[::3]
        #     np.savez(self.save_path + npz_.split('/')[-1], aud=aud_re, label=label)
        #
        #
        #     # For checking
        #     # plt.subplot(2,1 , 1)
        #     # plt.plot(aud_re)
        #     # plt.subplot(2, 1, 2)
        #     # plt.plot(label[::3])
        #     # plt.show()
        #     # exit()

        _ = parmap.map(self.resample_aud_label, npz_list, pm_pbar=True, pm_processes=24)

    def split_sound_tr_cv_tt_210301(self):

        sound_list = ['knock', 'speech', 'dog', 'clap']
        split_dict = dict()
        for sound in sound_list:
            # split_dict[sound]
            tr_cv_tt = dict()
            s_list = glob(self.save_path + '*{}*'.format(sound))
            s_list.sort()

            # split 8:1:1
            ntr = int(np.ceil(len(s_list)*0.85))
            ncv_tt = len(s_list)-ntr
            tr_list = s_list[:ntr]

            # rest of the list are used for validation and test list
            cv_tt_list = s_list[ntr:]
            np.random.shuffle(cv_tt_list)
            n_cv = int(ncv_tt/2)
            cv_list = cv_tt_list[:n_cv]
            tt_list = cv_tt_list[n_cv:]

            print('number of sound list=> tr: {}, cv: {}, tt: {}'.format(len(tr_list), len(cv_list), len(tt_list)))
            tr_cv_tt['tr'] = tr_list
            tr_cv_tt['cv'] = cv_list
            tr_cv_tt['tt'] = tt_list
            split_dict[sound] = tr_cv_tt
            print('h')

        np.save('source_tr_cv_tt_list.npy', split_dict)

    def copy_210301(self):
        # read
        tr_cv_tt_list = np.load('/media/jeonghwan/HDD2/IS2021/code/data/source_tr_cv_tt_list.npy', allow_pickle=True).item()

        for key, value in tr_cv_tt_list.items():
            print(key)
            print(value)

            for file in value['tr']:
                shutil.copyfile(file, file.replace(self.save_path, '/media/jeonghwan/HDD2/IS2021/dataset/sound_source/tr/'))

            for file in value['cv']:
                shutil.copyfile(file, file.replace(self.save_path, '/media/jeonghwan/HDD2/IS2021/dataset/sound_source/cv/'))

            for file in value['tt']:
                shutil.copyfile(file, file.replace(self.save_path, '/media/jeonghwan/HDD2/IS2021/dataset/sound_source/tt/'))

    def mix_rir_and_sound_source_210301(self):
        ### --------- single source dataset----------- ###
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/SSL/tt/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/tr/anechoic/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_test/'

        rir_list = glob(rir_path + '*.npz')
        spc_list = glob(spc_path + '*.wav')

        # generate random rir index
        spc_list.sort()
        for i, _spc in enumerate(tqdm(spc_list)):

            # read audio file
            aud, fs = librosa.core.load(_spc, sr=None, mono=False)

            if len(aud.shape) != 1:
                aud = aud[:, 0]

            #aud.shape[1]
            idx_s = np.random.randint(0, len(rir_list))
            npz = np.load(rir_list[idx_s], allow_pickle=True)

            # convolve
            rir = npz['rir']
            Y = ss.convolve(rir, aud[:, np.newaxis])
            audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.')[0]+'_'+_spc.split('/')[-1], Y, fs)
        # make configuration file and mix as it
        # pick random sound source
        # add random SNR mix

        # After mixing then
        # all sound source and rir will be convolved

        # RIR for training dataset

        # RIR for validation/test dataset

    def mix_rir_and_sound_source_210311(self):
        ### --------- single source dataset----------- ###
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Real_DB_SSLR/SSL/tr/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Real_DB_SSLR/simulated_RIR/anechoic/tr/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_train/'

        rir_list = glob(rir_path + '*.npz')
        spc_list = glob(spc_path + '*.wav')

        # generate random rir index
        spc_list.sort()
        _use_par = True

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir, spc_list, pm_pbar=True,  pm_processes=28)
        else:
            for i, _spc in enumerate(tqdm(spc_list)):

                # read audio file
                aud, fs = librosa.core.load(_spc, sr=None, mono=False)

                if len(aud.shape) != 1:
                    aud = aud[:, 0]

                #aud.shape[1]
                idx_s = np.random.randint(0, len(rir_list))
                npz = np.load(rir_list[idx_s], allow_pickle=True)

                # convolve
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.')[0]+'_'+_spc.split('/')[-1], Y, fs)
        # make configuration file and mix as it
        # pick random sound source
        # add random SNR mix

        # After mixing then
        # all sound source and rir will be convolved

        # RIR for training dataset

        # RIR for validation/test dataset

    def mix_rir_and_sound_source_210312(self):
        ### --------- single source dataset----------- ###
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Real_DB_SSLR/SSL/tt/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Real_DB_SSLR/simulated_RIR/anechoic/tr/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_test/'

        rir_list = glob(rir_path + '*.npz')
        spc_list = glob(spc_path + '*.wav')

        # generate random rir index
        spc_list.sort()
        _use_par = False

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir, spc_list, pm_pbar=True,  pm_processes=28)
        else:
            for i, _spc in enumerate(tqdm(spc_list)):

                # read audio file
                aud, fs = librosa.core.load(_spc, sr=None, mono=False)

                if len(aud.shape) != 1:
                    aud = aud[:, 0]

                #aud.shape[1]
                idx_s = np.random.randint(0, len(rir_list))
                npz = np.load(rir_list[idx_s], allow_pickle=True)

                # convolve
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.')[0]+'_'+_spc.split('/')[-1], Y, fs)
        # make configuration file and mix as it
        # pick random sound source
        # add random SNR mix

        # After mixing then
        # all sound source and rir will be convolved

        # RIR for training dataset

        # RIR for validation/test dataset

    def mix_rir_and_sound_source_210313(self):
        ### --------- single source dataset----------- ###
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tt/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/anechoic/tr/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_test/'

        rir_list = glob(rir_path + '*.npz')
        spc_list = glob(spc_path + '*.wav')

        # generate random rir index
        spc_list.sort()
        _use_par = True

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir, spc_list, pm_pbar=True,  pm_processes=28)
        else:
            for i, _spc in enumerate(tqdm(spc_list)):

                # read audio file
                aud, fs = librosa.core.load(_spc, sr=None, mono=False)

                if len(aud.shape) != 1:
                    aud = aud[:, 0]

                #aud.shape[1]
                idx_s = np.random.randint(0, len(rir_list))
                npz = np.load(rir_list[idx_s], allow_pickle=True)

                # convolve
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.')[0]+'_'+_spc.split('/')[-1], Y, fs)
        # make configuration file and mix as it
        # pick random sound source
        # add random SNR mix

        # After mixing then
        # all sound source and rir will be convolved

        # RIR for training dataset

        # RIR for validation/test dataset

    def mix_rir_and_sound_source_210313(self):
        ### --------- single source dataset----------- ###
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tt/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/anechoic/tr/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_test/'

        rir_list = glob(rir_path + '*.npz')
        spc_list = glob(spc_path + '*.wav')

        # generate random rir index
        spc_list.sort()
        _use_par = True

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir, spc_list, pm_pbar=True,  pm_processes=28)
        else:
            for i, _spc in enumerate(tqdm(spc_list)):

                # read audio file
                aud, fs = librosa.core.load(_spc, sr=None, mono=False)

                if len(aud.shape) != 1:
                    aud = aud[:, 0]

                #aud.shape[1]
                idx_s = np.random.randint(0, len(rir_list))
                npz = np.load(rir_list[idx_s], allow_pickle=True)

                # convolve
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.')[0]+'_'+_spc.split('/')[-1], Y, fs)
        # make configuration file and mix as it
        # pick random sound source
        # add random SNR mix

        # After mixing then
        # all sound source and rir will be convolved

        # RIR for training dataset

        # RIR for validation/test dataset

    def mix_rir_and_sound_source_210319(self):
        ### --------- single source dataset----------- ###
        ### generate tr ###
        # rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/'
        # spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_train/'
        ### generate tt ###
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tt/'
        spc_path = '/media/jeonghwan/HDD1/Dataset/MS-SNSD/clean_test/'

        rir_list = glob(rir_path + '**/*.npz')
        np.random.shuffle(rir_list) # random shuffle
        spc_list = glob(spc_path + '*.wav')
        num_spc = 3000 # for test
        spc_list *= int(num_spc / len(spc_list))

        # take random rir list len(spc_list)
        rir_idxs = np.random.randint(0, len(rir_list), len(spc_list))
        sampled_rir_list = [rir_list[idx] for idx in rir_idxs]

        # make speech-rir pair list for [[spc1, rir1], [spc2, rir2], ..., [spcN, rirN],]
        spc_rir_list = [[spc, sampled_rir_list[i]] for i, spc in enumerate(spc_list)]

        # generate random rir index
        _use_par = True

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir_mp, spc_rir_list, pm_pbar=True,  pm_processes=28)

        ### generate tt ###

    def mix_rir_and_noise_source_210319(self):
        ### --------- single source dataset----------- ###
        ### generate tr ###
        # rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/'
        # noi_path = '/media/jeonghwan/HDD2/IS2021/dataset/MS-SNSD_noise/noise_train/'

        ### generate tt ###
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tt/'
        noi_path = '/media/jeonghwan/HDD2/IS2021/dataset/MS-SNSD_noise/noise_test/'

        rir_list = glob(rir_path + '**/*.npz')
        np.random.shuffle(rir_list) # random shuffle
        noi_list = glob(noi_path + '*.wav')
        #num_noi=5000 # for trianing
        num_noi = 2000 # for test

        noi_list *= int(num_noi / len(noi_list))

        # take random rir list len(spc_list)
        rir_idxs = np.random.randint(0, len(rir_list), len(noi_list))
        sampled_rir_list = [rir_list[idx] for idx in rir_idxs]

        # make speech-rir pair list for [[spc1, rir1], [spc2, rir2], ..., [spcN, rirN],]
        spc_rir_list = [[spc, sampled_rir_list[i]] for i, spc in enumerate(noi_list)]

        # generate random rir index
        _use_par = True

        if _use_par == True:
            _ = parmap.map(self.convolve_and_save_rir_mp, spc_rir_list, pm_pbar=True,  pm_processes=28)

        ### generate tt ###

    def convolve_and_save_rir(self, fn):
        # read audio file
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tt/'
        rir_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/anechoic/tr/'
        mix_all = False
        rir_list = glob(rir_path + '*.npz')
        aud, fs = librosa.core.load(fn, sr=None, mono=False)

        if len(aud.shape) != 1:
            aud = aud[:, 0]
        if mix_all == True:
            for i, _rir in enumerate(rir_list):
                npz = np.load(_rir, allow_pickle=True)
                rir = npz['rir']
                Y = ss.convolve(rir, aud[:, np.newaxis])
                audiowrite(save_path+_rir.split('/')[-1].split('.')[0]+'_'+fn.split('/')[-1], Y, fs)
        else:
            idx_s = np.random.randint(0, len(rir_list))
            npz = np.load(rir_list[idx_s], allow_pickle=True)

            # convolve
            rir = npz['rir']
            Y = ss.convolve(rir, aud[:, np.newaxis])
            audiowrite(save_path+rir_list[idx_s].split('/')[-1].split('.n')[0]+'_'+fn.split('/')[-1], Y, fs)

    def convolve_and_save_rir_mp(self, fn):
        # read audio file
        # save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tr/'
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tt/'
        #save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_noisy_direct/tr/'
        #save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_noisy_direct/tt/'
        aud, fs = librosa.core.load(fn[0], sr=None, mono=False)
        if len(aud.shape) != 1:
            aud = aud[:, 0]
        room_num = fn[1].split('/')[-2]
        # convolve
        npz = np.load(fn[1], allow_pickle=True)
        rir = npz['rir']
        Y = ss.convolve(rir, aud[:, np.newaxis])
        audiowrite(save_path+fn[1].split('/')[-1].split('.n')[0]+ '_' + room_num + '_'+fn[0].split('/')[-1], Y, fs)

    def resample_aud_label(self, npz_):
        npz = np.load(npz_, allow_pickle=True)
        aud = npz['aud']
        label = npz['label']

        # downsample
        aud_re = librosa.resample(aud, 48000, 16000) # resampling

        # label downsample
        label = label[::3]

        np.savez(self.save_path + npz_.split('/')[-1], aud=aud_re, label=label)

    def run_210301(self):
            # 1
            self.preproce_db_210301()
            self.split_sound_tr_cv_tt_210301()
            self.copy_210301()
            self.mix_rir_and_sound_source_210301()
            self.add_diffuse_noise_variant_SNR()

    def run_210302(self):
            # 1
            self.mix_rir_and_sound_source_210301()
            #2 make VAD

    def run_210313(self):
            self.mix_rir_and_sound_source_210313()

    def split_tr_tt_diffuse(self):
        path_DEMAND = '/media/jeonghwan/HDD2/Dataset/DEMAND/'
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/diffuse_noise/tr/'

        # Channel1을 선택해서 diffuse로 만들었음
        noise_list = glob(path_DEMAND + '**/*ch01.wav')
        for i, noi in enumerate(noise_list):
            aud, fs = librosa.core.load(noi, sr=None, mono=False)
            fn = noi.split('/')[-2] + '_' + noi.split('/')[-1]

            len_tr = int(aud.shape[0]*4/5)
            noi_tr = aud[:len_tr]
            noi_tt = aud[len_tr:]

            audiowrite(save_path+fn.split('/')[-1], noi_tr, fs)
            audiowrite(save_path.replace('/tr/', '/tt/')+fn.split('/')[-1], noi_tt, fs)

    def split_tr_tt_demand(self):
        path_DEMAND = '/media/jeonghwan/HDD2/Dataset/DEMAND/'
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/demand_noise/tr/'

        # Channel1을 선택해서 diffuse로 만들었음
        noise_list = glob(path_DEMAND + '**/*ch01.wav')
        for i, noi in enumerate(noise_list):
            aud, fs = librosa.core.load(noi, sr=None, mono=False)
            aud = aud[np.newaxis, :]
            for j in range(1, 4):
                aud_temp, _ = librosa.core.load(noi.replace('ch01', 'ch0{}'.format(j+1)), sr=None, mono=False)
                aud = np.concatenate((aud, aud_temp[np.newaxis, :]), axis=0)

            fn = noi.split('/')[-2] + '_' + noi.split('/')[-1]

            len_tr = int(aud.shape[1]*4/5)
            noi_tr = aud[:, :len_tr]
            noi_tt = aud[:, len_tr:]

            audiowrite(save_path+fn.split('/')[-1], noi_tr.T, fs)
            audiowrite(save_path.replace('/tr/', '/tt/')+fn.split('/')[-1], noi_tt.T, fs)

class acoustic_simulator():

    def __init__ (self):
        print('acoustic simulotor v1')
        #self.initialize_room_params()
        self.save_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/16kHz/anechoic/'

    def get_config_for_R1(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [6, 6, 2.5]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        params['reverberation_time'] = 0.0
        params['nsample'] = 8192
        return params

    def get_config_for_R2(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [8, 6, 2.5]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        params['reverberation_time'] = 0.2
        params['nsample'] = 8192
        return params

    def get_config_for_R3(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [6, 8, 2.5]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        params['reverberation_time'] = 0.4
        params['nsample'] = 8192
        return params

    def get_config_for_R4(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [7, 5, 3]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        params['reverberation_time'] = 0.12
        params['nsample'] = 8192
        return params

    def get_config_for_R5(self):
        params = dict()
        params['c'] = 343
        params['fs'] = 16000
        params['L'] = [5, 7, 3]
        params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        params['s'] = []  # [2, 3.sgdgopdvd5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        params['reverberation_time'] = 0.32
        params['nsample'] = 8192
        return params

    def initialize_room_params(self):
        self.params = dict()
        self.params['c'] = 343
        self.params['fs'] = 16000
        self.params['L'] = [8, 8, 6]
        self.params['r'] = []  # self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True)  # [[], [], [], []]
        self.params['s'] = []  # [2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
        self.params['reverberation_time'] = 0.0
        self.params['nsample'] = 8192

        self.radius_mic = 0.04
        self.radius_src = 2
        self.nmic = 4
        self.resolution = [10, 10] # azimuth/elevation angle resolution,

        # for visual plotting
        self.visualization = False
        self.dim = '3D'

    def example(self):
        #signal, fs = sf.read("bark.wav", always_2d=True)
        params = self.generate_room_params([1, 1, 1])
        print(params)
        h = rir.generate(**params)
        plt.figure(1)
        plt.plot(h)
        plt.show()
        # visualization room configure
        # RIR 저장할 때 room config랑 같이 저장 할 것
        # print(signal.shape)  # (11462, 2)
        # Convolve 2-channel signal with 3 impulse responses
        # return signal=ss.convolve(h[:, None, :], signal[:, :, None])
        # print(signal.shape)  # (15557, 2, 3)

    def generate_rir_and_save(self):

        # 만들 때 room parameter를 print해서 보여 줄 것
        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = self.get_UCA_array(center=center, radius=self.radius_mic, nmic=self.nmic, visualization=self.visualization)
        self.params['r'] = mic_pos

        # create source position list
        src_pos_list = self.get_uniform_dist_circ_pos(center=center, radius=self.radius_src, resolution=self.resolution, dim=self.dim)

        # for i, src_pos in enumerate(src_pos_list):
        #     self.params['s'] = src_pos
        #     azi, ele, r = self.cart2sph(src_pos-center)
        #     h = rir.generate(**self.params)
        #     np.savez(self.save_path + 'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

        _ = parmap.map(self.generate_rir, src_pos_list, pm_pbar=True, pm_processes=24)

    def generate_rir(self, src_pos):
        self.params['s'] = src_pos
        center = (np.array(self.params['L']) / 2).tolist()
        azi, ele, r = self.cart2sph(src_pos - center)
        h = rir.generate(**self.params)
        np.savez(self.save_path + 'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    # def generate_room_params(self, source_position):
    #     params = dict()
    #     params['c'] = 343
    #     params['fs'] = 22050
    #     params['L'] = [5, 4, 6]
    #     center = (np.array(params['L'])/2).tolist()
    #     params['r'] = self.get_UCA_array(center=center, r=0.04, nmic=4, visualization=True) # [[], [], [], []]
    #     params['s'] = source_position #[2, 3.5, 2] # self.get_src_pos(center=[], n_resol=)#[2, 3.5, 2]
    #     params['reverberation_time'] = 0.0
    #     params['nsample'] = 8192
    #
    #     return params

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
            azi_list = np.arange(180, 181, resolution)
            ele_list = [0]

        for ele in ele_list:
            for azi in azi_list:
                pos_mics.append(center + self.sph2cart(radius, azi, ele))

        #for i in range(nmic):
        #    pos_mics.append(center + np.round(np.array([r*np.cos(mic_resol[i]*np.pi/180), r*np.sin(mic_resol[i]*np.pi/180), 0]), 3))

        #self.visualize_pos(pos_mics)

        return pos_mics

    def get_ULA_array(self, center, interspace, nmic):
        """
        첫번째 마이크
        [y, z] is awlays zero
        [x, 0, 0] cm
        interspace: 0.01[cm]
        :return: 마이크 위치 [cm]
        """
        maximum_distance = interspace * nmic
        mic_xs = [interspace*i for i in range(nmic)]

        # set center has zero value
        mic_xs = mic_xs - np.mean(mic_xs)
        pos_mics = []

        for i in range(nmic):
            pos_mics.append([mic_xs[i], 0, 0])

        return np.array([center]) + pos_mics

    def get_UCA_array(self, center, radius, nmic, visualization):
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
        rec_list = np.array(rec_list) # [ndata, (x,y,z)]
        #src_list = np.array(src_list)

        # ## Interactive visualization using plotly
        # fig = px.scatter_3d(x=rec_list[:, 0], y=rec_list[:, 1], z=rec_list[:, 2])
        # fig.show()
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
        # fig.add_trace(go.Scatter3d(x=src_list[:, 0],
        #                            y=src_list[:, 1],
        #                            z=src_list[:, 2],
        #                            mode="markers",
        #                            marker=dict(
        #                                color='blue',
        #                                size=2
        #                            )
        #                            ))
        # fig.update_layout(
        #     autosize=False,
        #     width=700,
        #     height=700,
        #     margin=go.layout.Margin(l=50, r=50, b=50,
        #                             t=50, pad=4),
        #     paper_bgcolor="white",
        # )
        #
        # # plot label in embedding space
        # camera = dict(up=dict(x=0, y=0, z=1),
        #               center=dict(x=0, y=0, z=0),
        #               eye=dict(x=-1.5, y=-1.5, z=1.5))
        #
        # fig.update_layout(scene_camera=camera,
        #                   scene=dict(xaxis=dict(range=[-2, 2]),
        #                              yaxis=dict(range=[-2, 2]),
        #                              zaxis=dict(range=[-2, 2]), ))
        fig.show()

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

    def cart2sph(self, position):
        x, y, z = position
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)

        # rad2deg
        az = np.round(az*180/np.pi, 0)
        el = np.round(el*180/np.pi, 0)

        if az < 0:
            az = np.abs(az) + 180

        return az, el, np.round(r, 1)

    def run(self):
        # generate simple multi-channel RIR, located microphone array specific location
        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = self.get_ULA_array(center=center, interspace=0.02, nmic=4)
        print(mic_pos)
        # mic_pos = self.get_UCA_array(center=center, radius=self.radius_mic, nmic=self.nmic, visualization=self.visualization)
        self.params['r'] = mic_pos

        # create source position list
        src_pos_list = self.get_uniform_dist_circ_pos(center=center, radius=self.radius_src, resolution=30, dim='2D')
        #save_path = ''
        for i, src_pos in enumerate(src_pos_list):
            self.params['s'] = src_pos
            azi, ele, r = self.cart2sph(src_pos-center)
            h = rir.generate(**self.params)
            np.savez('az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def run_210301(self):
        # generate simple multi-channel RIR, located microphone array specific location
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/tr/anechoic/'
        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = self.get_ULA_array(center=center, interspace=0.05, nmic=2)
        print(mic_pos)
        # mic_pos = self.get_UCA_array(center=center, radius=self.radius_mic, nmic=self.nmic, visualization=self.visualization)
        self.params['r'] = mic_pos

        # create source position list
        src_pos_list = self.get_uniform_dist_circ_pos(center=center, radius=self.radius_src, resolution=5, dim='2D')
        #save_path = ''
        for i, src_pos in enumerate(src_pos_list):
            self.params['s'] = src_pos
            azi, ele, r = self.cart2sph(src_pos-center)
            h = rir.generate(**self.params)
            np.savez(save_path+'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def run_210311(self):
        # goal: create new dataset for NS
        # 4-microphone array
        # generate simple multi-channel RIR, located microphone array specific location
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/SSLR/simulated_RIR/anechoic/tr/'
        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = np.array([[-0.0267, 0.0343, 0],
                            [-0.0267, -0.0343, 0],
                            [0.0313, 0.0343, 0],
                            [0.0313, -0.0343, 0]])
        print(mic_pos)
        self.params['r'] = mic_pos + center

        # create source position list
        src_pos_list = self.get_uniform_dist_circ_pos(center=center, radius=self.radius_src, resolution=[5, 5], dim='3D')

        for i, src_pos in enumerate(src_pos_list):
            self.params['s'] = src_pos
            azi, ele, r = self.cart2sph(src_pos-center)
            h = rir.generate(**self.params)
            np.savez(save_path+'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def run_210313(self):
        # goal: create new dataset for NS
        # 4-microphone array
        # generate simple multi-channel RIR, located microphone array specific location
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/SSLR/simulated_RIR/anechoic/tr/'
        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = np.array([[-0.0267, 0.0343, 0],
                            [-0.0267, -0.0343, 0],
                            [0.0313, 0.0343, 0],
                            [0.0313, -0.0343, 0]])
        print(mic_pos)
        self.params['r'] = mic_pos + center

        # create source position list
        src_pos_list = self.get_uniform_dist_circ_pos(center=center, radius=self.radius_src, resolution=[5, 5], dim='3D')

        for i, src_pos in enumerate(src_pos_list):
            self.params['s'] = src_pos
            azi, ele, r = self.cart2sph(src_pos-center)
            h = rir.generate(**self.params)
            np.savez(save_path+'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def run_210313_ULA_sim(self):
        # goal: create new dataset for NS
        # 4-microphone array
        # generate simple multi-channel RIR, located microphone array specific locations
        _use_par = True
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/anechoic/tr/'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        center = (np.array(self.params['L']) / 2).tolist()
        mic_pos = self.get_ULA_array(center, 0.05, 4)
        print(mic_pos)
        self.params['r'] = mic_pos

        # create source position list
        src_pos_list = []
        for r in np.linspace(1, 3, 21):
            src_pos_list += self.get_uniform_dist_circ_pos(center=center, radius=r, resolution=1, dim='2D')

        if _use_par == True:
            _ = parmap.map(self.generate_rir, src_pos_list)
        else:
            for i, src_pos in enumerate(src_pos_list):
                self.params['s'] = src_pos
                azi, ele, r = self.cart2sph(src_pos-center)
                h = rir.generate(**self.params)
                np.savez(save_path+'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def generate_rir(self, src_pos):
        center = (np.array(self.params['L']) / 2).tolist()

        self.params['s'] = src_pos
        azi, ele, r = self.cart2sph(src_pos-center)
        h = rir.generate(**self.params)
        np.savez(self.save_path+'az{}_el{}_r{}.npz'.format(int(azi), int(ele), r), rir=h, params=self.params)

    def create_rir_final_0318(self):
        _use_par = True

        # Create RIR for train (R1)
        params = self.get_config_for_R1()
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R1/'
        self.generate_rir_for_room(params, save_path)

        # Create RIR for train (R2)
        params = self.get_config_for_R2()
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R2/'
        self.generate_rir_for_room(params, save_path)

        # Create RIR for train (R3)
        params = self.get_config_for_R3()
        save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R3/'
        self.generate_rir_for_room(params, save_path)

        # Create RIR for train (R4)
        # params = self.get_config_for_R4()
        # save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tt/R4/'
        # self.generate_rir_for_room(params, save_path)

        # Create RIR for train (R5)
        # params = self.get_config_for_R5()
        # save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tt/R5/'
        # self.generate_rir_for_room(params, save_path)


    def generate_rir_for_room(self, params, save_path):
        # Create RIR for train (R1)
        #save_path = '/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/simulated_RIR/tr/R1/'
        _use_par = True

        Path(save_path).mkdir(parents=True, exist_ok=True)
        #params = self.get_config_for_R1()

        center = (np.array(params['L']) / 2).tolist()
        mic_pos = self.get_ULA_array(center, 0.05, 4)
        print(mic_pos)
        params['r'] = mic_pos
        self.params = params
        self.save_path = save_path
        # create source position list
        src_pos_list = []

        #for r in np.linspace(1.1, 2.3, 5): # for test
        for r in np.linspace(1, 3, 11): # for training

            src_pos_list += self.get_uniform_dist_circ_pos(center=center, radius=r, resolution=1, dim='2D')

        if _use_par == True:
            _ = parmap.map(self.generate_rir, src_pos_list)

class clean_source_mixer():

    def __init__(self):
        noise_path='h'
        clean_path='hh'
        self.fs = 16000
        self.feat_cls = feature_extractor()
        self.eps = np.finfo(np.float32).eps

    def generate_single_source(self):

        # read audio file
        aud, fs = audioread(path='wn.wav', sr=None, mono=False)

        # load rir file
        npz_path = './'
        fn = 'az150_el0_r2.0'
        npz_list = glob(npz_path + '{}.npz'.format(fn))

        npz = np.load(npz_list[0], allow_pickle=True)
        rir = npz['rir']

        Y = self.convolve_rir_signal(rir, aud)

        # save
        audiowrite('{}.wav'.format(fn), Y, 16000)
        return Y

    def mix_w_noise(self, azi, ele, r):
        # read audio file
        aud, fs = audioread(path='wn.wav', sr=None, mono=False)

        # load rir file
        npz_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/anechoic/'
        npz_list = glob(npz_path + 'az{}_el{}_r{}.npz'.format(azi, ele, r))

        npz = np.load(npz_list[0], allow_pickle=True)
        rir = npz['rir']

        Y = self.convolve_rir_signal(rir, aud)

        # add spatially uncorrelated white noise

        swn = np.random.randn(Y.shape[0], 4)-0.5
        Y += swn*0.002

        return Y

    def mix_spatially_white_noise(self, fn, SNR):
        """
        :param fn: audio filename
        :param SNR: want to mix
        :return:
        """

        # add spatially uncorrelated white noise

        swn = np.random.randn(Y.shape[0], 4)-0.5
        Y += swn*0.002

        audiowrite('mixed_{}dB.wav'.format(SNR))

        return

    def mix_2_sources(self, azi_list, ele_list, r_list):
        # read audio file
        aud, fs = audioread(path='wn.wav', sr=None, mono=False)

        # load rir file
        npz_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/anechoic/'
        npz_list1 = glob(npz_path + 'az{}_el{}_r{}.npz'.format(azi_list[0], ele_list[0], r_list[0]))
        npz_list2 = glob(npz_path + 'az{}_el{}_r{}.npz'.format(azi_list[1], ele_list[1], r_list[1]))
        npz1 = np.load(npz_list1[0], allow_pickle=True)
        npz2 = np.load(npz_list2[0], allow_pickle=True)

        rir1 = npz1['rir']
        rir2 = npz2['rir']

        X1 = self.convolve_rir_signal(rir1, aud)
        X2 = self.convolve_rir_signal(rir2, aud)

        # add spatially uncorrelated white noise

        #swn = np.random.randn(Y.shape[0], 4) - 0.5
        # Y += swn * 0.002

        # 두개의 signal을 더할 때도 SNR을 맞춰서 더해주어야 함
        # 데이터가 엄청나게 많이 필요하네

        return X1+2*X2, X1, X2

    def mix_2_sources_w_noise(self, azi_list, ele_list, r_list):
        # read audio file
        aud, fs = audioread(path='wn.wav', sr=None, mono=False)

        # load rir file
        npz_path = '/media/jeonghwan/HDD2/IS2021/dataset/simulated_RIR/anechoic/'
        npz_list1 = glob(npz_path + 'az{}_el{}_r{}.npz'.format(azi_list[0], ele_list[0], r_list[0]))
        npz_list2 = glob(npz_path + 'az{}_el{}_r{}.npz'.format(azi_list[1], ele_list[1], r_list[1]))
        npz1 = np.load(npz_list1[0], allow_pickle=True)
        npz2 = np.load(npz_list2[0], allow_pickle=True)

        rir1 = npz1['rir']
        rir2 = npz2['rir']

        X1 = self.convolve_rir_signal(rir1, aud)
        X2 = self.convolve_rir_signal(rir2, aud)

        # add spatially uncorrelated white noise

        # swn = np.random.randn(Y.shape[0], 4) - 0.5
        # Y += swn * 0.002

        # 두개의 signal을 더할 때도 SNR을 맞춰서 더해주어야 함
        # 데이터가 엄청나게 많이 필요하네

        return X1 + 2 * X2, X1, X2

    def _adjusting_length(self, clean, noise, snr):
        """
        :param clean:
        :param noise:
        :param snr:
        :return:
        """
        # adjust to equal sample length
        clean_length = clean.shape[0]
        noise_length = noise.shape[0]
        st = 0  # choose the first point
        # padding the noise
        if clean_length > noise_length:
            # st = numpy.random.randint(clean_length + 1 - noise_length)
            noise_t = np.zeros([clean_length])
            noise_t[st:st+noise_length] = noise
            noise = noise_t
        # split the noise
        elif clean_length < noise_length:
            # st = numpy.random.randint(noise_l)
            print('ho')
        return self.snr_mix(clean, noise, snr)

    def snr_mix(self, clean, noise, snr):
        '''
        mix clean and noise according to snr
        clean: [nsamples,]
        noise: [nsamples,]
        snr: (float)
        '''
        #

        clean_rms = self._calculate_rms(clean)
        clean_rms = np.maximum(clean_rms, self.eps)
        noise_rms = self._calculate_rmsrms(noise)
        noise_rms = np.maximum(noise_rms, self.eps)
        k = np.sqrt(clean_rms / (10**(snr/10) * noise_rms))
        new_noise = noise * k
        return clean + new_noise

    def _calculate_rms(self, signal):
        """
        calc rms of wav
        """
        energy = signal ** 2
        max_e = np.max(energy)
        low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB
        RMS = np.mean(energy[energy>=low_thres])

        return RMS

    def generate_whitenoise(self):
        wn = np.random.randn(self.fs*5, 1)-0.5 # create random signal [0, 1]
        wn = wn/np.max(np.abs(wn))*0.5
        plt.figure()
        plt.plot(wn)
        plt.show()
        audiowrite('wn.wav', wn, self.fs)
        return wn

    def convolve_rir_signal(self, rir, signal):
        print(rir.shape)    # [sample, channel](4096, 3)
        print(signal.shape) # [samaple, channel](11462, 1)
        signal = ss.convolve(rir, signal[:, np.newaxis])
        #plt.figure(1)
        #plt.plot(signal)
        #plt.show()
        return signal

    def run(self):

        # read single wav-file
        # generate room impulse response
        # ddd
        self.snr_mix()
        # generate white noise signal
        # self.generate_single_source()

        # rir generatorf
        #self.mix_w_noise()

        #For elevation 변동
        # for el in [0, 30, 60, 90]:
        #     Y = self.mix(0, el, 1.0)
        #     GCC_PHAT = self.feat_cls.get_GCCpattern_signal(Y, visualization=True, pooling='freq')
        #     self.feat_cls.visualize_GCC_PHAT(GCC_PHAT)

        # for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
        #     Y = self.mix(30, 0, r)
        #     GCC_PHAT = self.feat_cls.get_GCCpattern_signal(Y, visualization=True, pooling='freq')
        #     self.feat_cls.visualize_GCC_PHAT(GCC_PHAT)

        # 2mix

        # Y, X1, X2 = self.mix_2_sources([270, 0], [0, 60], [1.0, 1.0])
        #
        # for s in [Y, X1, X2]:
        #     GCC_PHAT = self.feat_cls.get_GCCpattern_signal(s, visualization=True, pooling='freq')
        #     self.feat_cls.visualize_GCC_PHAT(GCC_PHAT)

    def check_diffuse_noise(self):
        # add
        dir_path ='/home/jeonghwan/Downloads/TBUS/'
        aud = np.zeros((4, 160000))
        for i in range(4):
            aud_temp, fs = audioread(dir_path + 'ch0{}.wav'.format(i+1), sr=None, mono=False)
            aud[i, :] = aud_temp[fs*70:fs*80]

        GCC_PHAT = self.feat_cls.get_GCCpattern_signal(aud.T, visualization=True, pooling='freq')
        self.feat_cls.visualize_GCC_PHAT(GCC_PHAT)

    def generate_diffuse_noise(self):
        # using MATLAB
        print('You should use MATLAB script written by E. habet')

def create_experimental_dataset():

    # 1. generated simulated RIR
    AS = acoustic_simulator()
    AS.run_210301()

    # 2. Convolving

    # 3. Add diffuse noise (-5, 0, 5, 10 15) and spatially uncorrelated noise (30dB)
    # we used codes from MS-SNSD baseline.
    #mixer = clean_source_mixer()
    #mixer.run()

def create_sim_dataset_M4_CA():
    # 1. generated simulated RIR
    AS = acoustic_simulator()
    AS.run_210301()

def create_sim_dataset_M4_ULA():
    # 1. generated simulated RIR
    AS = acoustic_simulator()
    AS.run_210313_ULA_sim()

def create_sslr_NS_dataset():

    # 1. generated simulated RIR
    AS = acoustic_simulator()
    AS.run_210311()

    # 2. Convolving

def create_dataset_rir_final_0318():
    # 1. generated simulated RIR
    AS = acoustic_simulator()
    AS.create_rir_final_0318()

def check_statistics():
    wav_list = glob('/media/jeonghwan/HDD2/IS2021/dataset/Simul_DB_ULA4/multi_channel_speech/tr/*.wav')

if __name__=="__main__":
    #create_dataset_rir_final_0318()
    pd = preproc_dataset()
    pd.split_tr_tt_demand()
    #pd.mix_rir_and_noise_source_210319()
    #check_statistics()
    # 준비물
    # sound activity 구간이 표시된 음원
    # RIR data
    # diffuse noise generator
    # SSL: frame-level accuracy
    # AS = acoustic_simulator()
    # AS.run()
    # for different radius of source
    # for r in [0.5]:
    #     print('source radius: {}'.format(r))
    #     AS.radius_src = r
    #     AS.generate_rir_and_save() # self, center, radius, nresol
    # 잡음을 제거해야함
    # mixing part



