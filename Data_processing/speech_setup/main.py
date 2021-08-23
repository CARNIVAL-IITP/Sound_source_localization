import os, glob
import random
import pandas as pd
import soundfile as sf
import scipy
import pathlib

random.seed(777)

def main():
    data_dict={'audio_directory':[],'speaker':[], 'gender':[], 'VAD_directory':[], 'length':[]}
    # exit()
    
    dict02_dir='/data/Dataset/sitec/Dict02/clncut/'
    dict01_dir='/data/Dataset/sitec/Dict01/clncut/'
    data_dir_list=[dict01_dir, dict02_dir]
    vad_dir='/data/Dataset/sitec/VAD_by_sample/'
    vad_list_generate=pathlib.Path(vad_dir).rglob('*.mat')

    vad_dir_list=[str(i) for i in vad_list_generate]
    # print(vad_dir_list)
    # exit()
    # if 'abc' in 'abc123':
    #     print('hi')

   

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
                    # print(len(vad_dir_list))
            # print(data_dict)
    #     exit()
    # exit()
    pd.DataFrame(data_dict).to_csv('data.csv')

            # pri
    # all_dir=glob.glob(dict02_dir, recursive=True)
    # print(all_dir)
    exit()
    
    

    
    gender_list=['male', 'female']

    for dict_num in data_dir_list:
        for gender in gender_list:
            data_dir=dict_num+gender+'/'

            for speaker in os.listdir(data_dir):
                data_dir=data_dir+speaker+'/'

                for audio_dir in os.listdir(data_dir):
                    audio_dir=data_dir+audio_dir

    x=os.listdir(dict01_dir)
    pk=os.path.join(dict01_dir, x[0])
    print(pk)
    print(x)
    exit()

if __name__=='__main__':
    main()