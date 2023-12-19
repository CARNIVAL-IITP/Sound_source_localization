from glob import glob
import pandas as pd
import soundfile as sf
from soundfile import SoundFile, SEEK_END
import tqdm

class PreProcessor():
    def __init__(self):
        self.train_dir='/MS-SNSD/noise_train/'
        self.test_dir='/MS-SNSD/noise_test/'

        self.metadata_dir='./metadata/'
        train_file='ms-snsd_train.csv'
        test_file='ms-snsd_test.csv'

        self.make_csv(self.metadata_dir+train_file, self.train_dir)
        self.make_csv(self.metadata_dir+test_file, self.test_dir)

    def make_csv(self, save_dir, audio_dir):

        # dir + speaker_id + chaper_id + wav_name
        column_names=['file_path', 'duration', ]
        csv_dict={}

        for column in column_names:
            csv_dict[column]=[]


     
        csv_data=pd.DataFrame(columns=column_names)

        audio_list=glob(audio_dir+'/**/*.wav', recursive=True)

        for audio in tqdm.tqdm(audio_list, total=len(audio_list)):
            f=SoundFile(audio)

            wav_len = f.seek(0, SEEK_END)
            audio_name=audio.split('/')[-1]

            for column, data in zip(column_names, [audio_name, wav_len,]):
                csv_dict[column].append(data)
        
        pd.DataFrame(csv_dict).to_csv(save_dir)       



if __name__=='__main__':
    PreProcessor()
