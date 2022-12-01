import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve
from scipy.io.wavfile import write


rir='./circle_n8_az210_el0_r1.5.npz'
audio='./p229_008.wav'
rir=np.load(rir, allow_pickle=True)
rir=dict(zip(("data1{}".format(k) for k in rir), (rir[k] for k in rir)))['data1rir']*4

audio, sr=sf.read(audio)

result=[]
for i in range(rir.shape[-1]):
    data=oaconvolve(audio, rir[:,i], mode='same')
    result.append(data)
result=np.array(result).T*5
print(result.shape[1])
write('test.wav', sr, result)
# sf.write(result,'test.wav',  sr)