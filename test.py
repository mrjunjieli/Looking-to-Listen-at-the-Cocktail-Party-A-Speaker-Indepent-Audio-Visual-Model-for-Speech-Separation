import sys
sys.path.append ('./model/AV_model')
sys.path.append ('./data/utils')
from keras.models import load_model
from option import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import tensorflow as tf
import librosa

#parameters
people = 2
num_gpu=1

#path
model_path = './saved_AV_models/AVmodel-2p-065-0.40568.h5'
result_path = './predict/'
os.makedirs(result_path,exist_ok=True)

database = '/Work19/2020/lijunjie/LRS3/AV_model_database/mix/test/'
face_emb_path = '/Work19/2020/lijunjie/LRS3/face1022_emb/test/'


single_stft_path ='/Work19/2020/lijunjie/LRS3/AV_model_database/single/test/'
single_wav_path = '/CCALab/lijunjie/LRS3_process/audio_norm/test/'
single_wav_filename = ''


print('Initialing Parameters......')

#loading data
print('Loading data ......')
test_file = []
with open('/Work19/2020/lijunjie/LRS3/AV_model_database/AVdataset_test.txt','r') as f:
    test_file = f.readlines()


def get_data_name(line,people=people,database=database,face_emb=face_emb_path,single_stft_path=single_stft_path):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(people):
        single_idxs.append(names[i])
    file_path = database + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,1792,people))
    for i in range(people):
        face_embs[0,:,:,:,i] = np.load(face_emb+"%s_faceemb.npy"%single_idxs[i])

    single_stft = np.zeros((1,people,16000*3))
    for i in range(people):
        for file in os.listdir(os.path.join(single_stft_path,single_idxs[i])):
           single_wav_filename = 'norm_'+os.path.splitext(file.split('_')[1])[0]+'.wav'
           x,_ = librosa.load(os.path.join(single_wav_path,single_idxs[i],single_wav_filename),sr=16000)
           single_stft[0,i,:] = x[0:16000*3]

    return mix,single_idxs,face_embs,single_stft

#result predict
av_model = load_model(model_path,custom_objects={'tf':tf},compile=False)
if num_gpu>1:
    parallel = ModelMGPU(av_model,num_gpu)
    for line in test_file:
        mix,single_idxs,face_emb,single_stft = get_data_name(line,people,database,face_emb_path,single_stft_path)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = parallel.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(len(cRMs)):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            si_snr = utils.cal_si_snr(F,single_stft[:,:,:,i])
            print('si_snr:',si_snr)
            T = utils.fase_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)

if num_gpu<=1:
    for line in test_file:
        mix,single_idxs,face_emb,single_stft = get_data_name(line,people,database,face_emb_path,single_stft_path)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = av_model.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        T_=np.zeros((1,people,3*16000))
        for i in range(cRMs.shape[-1]):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            # # si_snr = utils.cal_si_snr(F,F)
            T = utils.fast_istft(F,power=False)
            T_[0,i,:] = T
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)
        si_snr,_,si_snr_idx = utils.cal_si_snr_with_pit(T_,single_stft)
        print('si_snr:',si_snr)
