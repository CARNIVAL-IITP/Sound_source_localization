from datetime import date
import os, glob
import random
import pandas as pd
from scipy.sparse import data
import soundfile as sf
import scipy
import pathlib
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy.signal import fir_filter_design, oaconvolve
import copy

random.seed(777)
np.random.seed(777)
# sklearn

def main():
    data_dict={'audio_directory':[],'speaker':[], 'gender':[], 'VAD_directory':[], 'length':[]}
    
    
    dict02_dir='/data/Dataset/sitec/Dict02/clncut/'
    dict01_dir='/data/Dataset/sitec/Dict01/clncut/'
    data_dir_list=[dict01_dir, dict02_dir]
    vad_dir='/data/Dataset/sitec/VAD_by_sample/'
    vad_list_generate=pathlib.Path(vad_dir).rglob('*.mat')

    vad_dir_list=[str(i) for i in vad_list_generate]

   

    for data_dir in data_dir_list:
        for audio_dir in pathlib.Path(data_dir).rglob('*.wav'):
            audio_dir=str(audio_dir)
            # print("HI")
            audio_info=audio_dir.split('/')
            speaker=audio_info[-2]
            gender=audio_info[-3]

            vad=audio_info[-1].replace('cut.wav', '.mat')

            for vad_file_dir in vad_dir_list:
                vad_split=vad_file_dir.split('/')
                if (vad == vad_split[-1]) and (speaker==vad_split[-2]):
                    
                    
                    data_dict['audio_directory'].append(audio_dir)
                    data_dict['speaker'].append(speaker)
                    data_dict['gender'].append(gender)
                    data_dict['VAD_directory'].append(vad_file_dir)
                    audio_file, fs= sf.read(audio_dir)
                    
                    data_dict['length'].append(audio_file.shape[0])
                    vad_dir_list.remove(vad_file_dir)
                    break

    pd.DataFrame(data_dict).to_csv('./audio_data.csv')


def rir_plot(data):
    data=np.load(data)['rir'][:,0]
    plt.plot(data)
    plt.show()
    print(data.shape)
    exit()


def rir_csv():
    rir_dir='/home/intern0/Desktop/project/IITP/Sound_source_localization/Data_processing/speech_setup/rir_gen/rir/'
    room_list=os.listdir(rir_dir)
    # data_dict={'rir_directory':[],'azi':[], 'ele':[], 'r':[], 'shape':[], 'mic_num':[], 'room':[]}
    train_dict={'rir_directory':[],'azi':[], 'ele':[], 'r':[], 'shape':[], 'mic_num':[], 'room':[]}
    cv_dict={'rir_directory':[],'azi':[], 'ele':[], 'r':[], 'shape':[], 'mic_num':[], 'room':[]}
    test_dict={'rir_directory':[],'azi':[], 'ele':[], 'r':[], 'shape':[], 'mic_num':[], 'room':[]}


    for room in room_list:
        direc=rir_dir+room+'/'
        # print(direc)
        rir_list=glob.glob(direc+'*.npz')

        if room in ['room_s1', 'room_409']:
            data_dict=test_dict
        else:
            
            data_dict=train_dict
        
        for rir in rir_list:
            data_dict['rir_directory'].append(rir)
            # print(rir)
            # exit()
            
            rir=rir.split('/')[-1][:-4].split('_')
            data_dict['shape'].append(rir[0])
            data_dict['azi'].append(rir[2][2:])
            data_dict['ele'].append(rir[3][2:])
            data_dict['r'].append(rir[4][1:])
            data_dict['mic_num'].append(rir[1][1:])
            data_dict['room'].append(room)
            # print(data_dict)
            # exit()


    
    test_dict=pd.DataFrame(test_dict)
    train_dict=pd.DataFrame(train_dict)
    pd.concat([test_dict, train_dict]).to_csv('rir_info.csv')

    condition ={'shape':['circle'], 'mic_num':[4]}
    df=test_dict.loc[test_dict['mic_num']=='4']
    test_dict=df.loc[df['shape']=='circle']
    
    test_dict.to_csv('./test_rir.csv')

    df=train_dict.loc[train_dict['mic_num']=='4']
    train_dict=df.loc[df['shape']=='circle']

    
    trcv=sklearn.utils.shuffle(train_dict)
    
    tr=trcv.iloc[:390,:]
    cv=trcv.iloc[390:,:]
    # np.random.shuffle(trcv.values)
    tr.to_csv('tr_rir.csv')
    cv.to_csv('cv_rir.csv')
    # exit()


def noise_list(audio_path, output_file, type):
    noise_type=[ 'AirConditioner', 'Babble', 'CopyMachine', 'ShuttingDoor', 'Typing']
    data_dict={'noise_directory':[],'type':[], 'length':[]}
    
    wav_list=glob.glob(audio_path+'*.wav')
    
    for wav in wav_list:
        audio_type=wav.split('/')[-1].split('_')[0]
        if audio_type in noise_type:
            data_dict['noise_directory'].append(wav)
            data_dict['type'].append(audio_type)
            file, fs=sf.read(wav)
            data_dict['length'].append(file.shape[0])

    if type=='tt':
        pd.DataFrame(data_dict).to_csv(output_file)
    else:
        train_df=pd.DataFrame()
        cv_df=pd.DataFrame()
        df=pd.DataFrame(data_dict)
        df=df.sort_values(by=['type'])
        # print(df)
        # exit()
        for noise_t in noise_type:
            noise_t_only=df[df['type']==noise_t]
            noise_t_only=sklearn.utils.shuffle(noise_t_only)
            train_df=pd.concat([train_df, noise_t_only[:-2]])
            cv_df=pd.concat([cv_df, noise_t_only[-2:]])
        
        train_df.to_csv(output_file[0])
        cv_df.to_csv(output_file[1])



def audio_split():
    data=pd.read_csv('./audio_data.csv')
    ori_speaker_list=data['speaker'].tolist()
    gender_list=data['gender'].tolist()
    length_list=data['length']
    newone=[]
    count=[]
    gengen=[]
    speaker_list=[]

    for speaker in ori_speaker_list:
        if speaker not in speaker_list:
            speaker_list.append(speaker)




    # speaker_list=list(set(ori_speaker_list))
    
    small_length_speaker=speaker_list[:400]
    
    large_length_speaker=speaker_list[400:]
    random.shuffle(small_length_speaker)

    test_speaker_list=small_length_speaker[:60]
    trcv_speaker_list=small_length_speaker[60:]+large_length_speaker

    tt_dataframe=[]
    trcv_dataframe=[]



    for num, speaker in enumerate(ori_speaker_list):
        ff=data.iloc[num]
        if speaker in test_speaker_list:
            
            tt_dataframe.append(ff)
        
        else:
            trcv_dataframe.append(ff)
            
            
    print(len(tt_dataframe))
    tt_dataframe=pd.DataFrame(tt_dataframe)
    trcv_dataframe=pd.DataFrame(trcv_dataframe)
    
    tt_dataframe.to_csv('./test_audio.csv')
    trcv_dataframe.to_csv('./trcv_audio.csv')

    trcv=sklearn.utils.shuffle(trcv_dataframe)
    cv_len=int(len(trcv)*0.9)

    tr_data=trcv.iloc[:cv_len,:]
    cv_data=trcv.iloc[cv_len:,:]
    np.random.shuffle(trcv.values)
    tr_data.to_csv('tr_audio.csv')
    cv_data.to_csv('cv_audio.csv')
    print(len(tr_data))
    print(len(cv_data))
    # exit()

def snr_count(speech, noise, snr):
    speech=np.abs(speech).sum()
    noise=np.abs(noise).sum()
    snr_mul=speech/noise/np.exp(snr/20)

    return snr_mul


def synthesize_reverb(mode):
    if mode=='tr':
        save_loc='./dataset/tr/'
        audio_csv=pd.read_csv('tr_audio.csv')
        rir_csv=pd.read_csv('tr_rir.csv')
        noise_csv=pd.read_csv('noise_train.csv')
        input_prefix='train_input_'
        label_prefix='train_label_'
        out_csv='tr.csv'


    elif mode=='cv':
        save_loc='./dataset/cv/'
        audio_csv=pd.read_csv('cv_audio.csv')
        rir_csv=pd.read_csv('cv_rir.csv')
        noise_csv=pd.read_csv('noise_cv.csv')
        input_prefix='cv_input_'
        label_prefix='cv_label_'
        out_csv='cv.csv'

    elif mode=='tt':
        save_loc='./dataset/tt/'
        audio_csv=pd.read_csv('test_audio.csv')
        rir_csv=pd.read_csv('./test_rir.csv')
        noise_csv=pd.read_csv('noise_test.csv')
        input_prefix='test_input_'
        label_prefix='test_label_'
        out_csv='tt.csv'

    del rir_csv['Unnamed: 0']
   
    audio_csv=sklearn.utils.shuffle(audio_csv)
    rir_csv=sklearn.utils.shuffle(rir_csv)
    noise_csv=sklearn.utils.shuffle(noise_csv)
    len_rir=len(rir_csv)
    len_noise=len(noise_csv)
    data_dict={'input_path':[], 'label_path':[], 'speech_path':[],'speech_RIR':[],'noise_path':[], 'noise_RIR':[], 'room':[], 'SNR':[], 'vad_path':[]}
    SNR_list=[0,5, 10,15]

    for num in range(len(audio_csv)):
        first=True
        # print(len(rir_csv))
        audio=audio_csv.iloc[num]
        rir=rir_csv.iloc[num%len_rir]
        # print(rir)
        # exit()

        noise=noise_csv.iloc[num%len_noise]
        
        audio_file, fs=sf.read(audio['audio_directory'])
        noise_file, fs=sf.read(noise['noise_directory'])

        
        audio_rir_file=np.load(rir['rir_directory'])['rir']

        room=rir['room']
        noise_rir=rir_csv.drop([num%len_rir], axis=0)   
        noise_rir=noise_rir.loc[rir_csv['room']==room]
        
     
             
        # print(noise_rir, room)
        noise_rir=noise_rir.sample()['rir_directory']
    
        noise_rir_file=np.load(noise_rir.item())['rir']

        
        input_name=save_loc+'input/'+input_prefix+str(num)+'.wav'
        output_name=save_loc+'label/'+label_prefix+str(num)+'.wav'

        data_dict['input_path'].append(input_name)
        data_dict['label_path'].append(output_name)
        data_dict['speech_path'].append(audio['audio_directory'])
        data_dict['speech_RIR'].append(rir['rir_directory'])
        data_dict['noise_path'].append(noise['noise_directory'])
        data_dict['noise_RIR'].append(noise_rir.item())
        data_dict['room'].append(room)
        data_dict['vad_path'].append(audio['VAD_directory'])
        SNR=random.choice(SNR_list)
        data_dict['SNR'].append(SNR)
        # print(SNR)
        # exit()
        long_noise=True
        if noise_file.shape[0]>audio_file.shape[0]:
            start=np.random.randint(0, noise_file.shape[0]-audio_file.shape[0])
            noise_file=noise_file[start:start+audio_file.shape[0],]
            
        
        else:
            pad_front=np.random.randint(0, audio_file.shape[0]-noise_file.shape[0])
            # print(pad_front)
            # pad_front=np.random.randint(0, audio_file.shape[0]-noise_file.shape[0])
            long_noise=False
        # print(audio_file.shape, noise_file.shape)
        # exit()

        
        for n in range(audio_rir_file.shape[-1]):
            
            audio_rired=oaconvolve(audio_file, audio_rir_file[:,n])
            noise_rired=oaconvolve(noise_file, noise_rir_file[:,n])
            if first==True:
                sf.write(output_name, audio_rired, fs)
                snr_mul=snr_count(audio_rired, noise_rired, SNR)
                result=np.zeros((audio_rired.shape[0], audio_rir_file.shape[-1]))
                first=False

            if long_noise==False:
                result[:, n]=audio_rired
                result[pad_front+noise_rired.shape[0], n]+=noise_rired*snr_mul
                
            else:
                result[:, n]=audio_rired+noise_rired*snr_mul
       
        sf.write(input_name, result, fs)
        # break
        # exit()
    pd.DataFrame(data_dict).to_csv('./dataset/'+out_csv)
    


if __name__=='__main__':  
    # main()
    # audio_split()
    # rir_csv()
    # test_path='/data/Dataset/MS-SNSD/noise_test/'
    # trcv_path='/data/Dataset/MS-SNSD/noise_train/'
    # # noise_list(test_path,'./noise_test.csv', 'tt' )
    # noise_list(trcv_path,['./noise_train.csv', './noise_cv.csv'], 'tr' )

    # exit()

    synthesize_reverb('tr')
    synthesize_reverb('cv')
    synthesize_reverb('tt')