clc; clear;

file='1/Audio Track-2.wav';
[audio, fs]=audioread(file);
% plot(audio)
% win = hamming(50e-3 * fs,'periodic');
% plot(audio)

spk1=[47178 208252; 227464 302345; 312963 379259];
spk2=[529471 817760];
spk3=[908636 1061060; 1098740 1228880];
spk4=[1354160 1498530; 1498530 1637440];
% size(spk1)
% spk1(1,1)
result=[];
spk=spk4;
shape=size(spk);
for col = 1:shape(1)
    fi=audio(spk(1,1):spk(shape(1),2));
    power=mean( fi.*fi);
    result=[result power];
end

result=mean(result);

final=10*log10(result);
final
% return
% power1=mean( fi.*fi)
% plot(fi)

res=[-92.7188 -104.1018 -101.7718 -89.6970];
res=res-res(1)
res=[-40.2673 -45.2108 -44.1990 -38.9549];
res=res-res(1)
res=[-40.8007 -45.2108 -44.5592 -38.9603];
res=res-median(res)
