import numpy as np
import soundfile as sf
import os
from scipy.signal import convolve, fftconvolve, oaconvolve
import matplotlib.pyplot as plt

speaker=[276, 274, 246, 244, 262, 260, 227, 225]
rir_list=['ellipsoid_n8_az120_el0_r1.5.npz', 'ellipsoid_n8_az270_el0_r1.0.npz', 'ellipsoid_n8_az60_el0_r0.5.npz', 'ellipsoid_n8_az0_el0_r1.5.npz', 'ellipsoid_n8_az300_el0_r1.5.npz', 'ellipsoid_n8_az210_el0_r1.5.npz', 'ellipsoid_n8_az180_el0_r1.0.npz']
total_data=[]
length=0
for wav, rir in zip(speaker, rir_list):
    wav_name='p'+str(wav)+'_004.wav'
    wav_data, fs=sf.read(wav_name)
    rir_data=np.load(rir, allow_pickle=True)
    rir_data=rir_data['rir']
    final=None
    
    for num in range(rir_data.shape[1]):
        result=oaconvolve(wav_data, rir_data[:,num])
        result=np.expand_dims(result, axis=-1)
        if num==0:
            final=result
        else:
        
            final=np.concatenate((final, result), axis=1)
    length+=final.shape[0]
    total_data.append(final)

white_rir='ellipsoid_n8_az240_el0_r1.5.npz'
white_rir=np.load(white_rir, allow_pickle=True)
white_rir=white_rir['rir']
white_noise = np.random.normal(0, 0.04, size=length)

for z in range(white_rir.shape[1]):
    result=oaconvolve(white_noise, white_rir[:,z])
    result=np.expand_dims(result, axis=-1)
    # print(result.shape)
    if z==0:
        final=result
    else:
    
        final=np.concatenate((final, result), axis=1)

dd=np.zeros((length, 8))+final[:length,:]
start=0
final=0
for i in total_data:
    dd[start:start+i.shape[0],:]+=i
    start=start+i.shape[0]-3*fs
sf.write('final2.wav', dd, fs)
# plt.plot(dd[:,0])
# plt.show()


    
    