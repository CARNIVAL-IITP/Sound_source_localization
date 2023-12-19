import webrtcvad
import pathlib
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os 
import matplotlib.pyplot as plt

def _cleanSilences(s, vad_tool, fs, aggressiveness, return_vad=False):
    vad_tool.set_mode(aggressiveness)

    vad_out = np.zeros_like(s)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(s) // vad_frame_len
    for frame_idx in range(n_vad_frames):
        frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = vad_tool.is_speech(frame_bytes, fs)
    
    s_clean = s * vad_out
    
    
    return (s_clean, vad_out) if return_vad else s_clean



vad_tool=webrtcvad.Vad()

wav_folder=dict()
wav_folder['train-clean-100']='/LibriSpeech/train-clean-100/'

vad_folder=dict()
vad_folder['train-clean-100']='/vad_by_webrtcpy_1005/train-clean-100/'

key_list=['train-clean-100']


for key in tqdm(key_list):

    vad_dir=vad_folder[key]
    data_dir=wav_folder[key]

    for audio_dir in tqdm(pathlib.Path(data_dir).rglob('*.wav')):
        audio_dir=str(audio_dir)
        audio_file, fs= sf.read(audio_dir)
        s_clean, vad_out=_cleanSilences(audio_file, vad_tool, fs, 3, return_vad=True)

        if np.count_nonzero(s_clean) < len(audio_file) * 0.66:
            
            s_clean, vad_out = _cleanSilences(audio_file, vad_tool, fs, 2, return_vad=True)
        if np.count_nonzero(s_clean) < len(audio_file) * 0.66:
            # print(3)
            s_clean, vad_out = _cleanSilences(audio_file, vad_tool, fs, 1, return_vad=True)

            
        vad_out=vad_out.astype(bool)

        vad_name=audio_dir.replace('.wav', '.npy')
        vad_name=vad_name.replace(data_dir, vad_dir)
        os.makedirs(os.path.dirname(vad_name), exist_ok=True)
        
        np.save(vad_name, vad_out)