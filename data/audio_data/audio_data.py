#author JunjieLi
#create 2020/11

import sys

sys.path.append('../utils')
import os
import librosa
import numpy as np
import utils
import itertools
import time
import random
import math
import scipy.io.wavfile as wavfile
import soundfile
import random

audio_norm_path = '/CDShare2/lijunjie/LRS3/audio_norm'
database_path = '/Work19/2020/lijunjie/LRS3/AV_model_database'
sample_rate = 16000
data_classes = ['pretrain','trainval','test']



num_speakers = 2
max_generate_data = 50


def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")


# initial data dir
def init_dir(path=database_path):
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists('%s/mix' % path):
        os.mkdir('%s/mix' % path)

    if not os.path.isdir('%s/single' % path):
        os.mkdir('%s/single' % path)

    if not os.path.isdir('%s/crm' % path):
        os.mkdir('%s/crm' % path)

    if not os.path.isdir('%s/mix_wav' % path):
        os.mkdir('%s/mix_wav' % path)
    # if not os.path.isdir('%s/single_joint_wav'%path):
    #     os.mkdir('%s/single_joint_wav'%path)


# def joint_single(single_input=audio_norm_path,data_path=database_path,data_classes=data_classes):


#     print('start joint single wav')

#     for data in data_classes:
#         mkdir(os.path.join(data_path,'single_joint_wav',data))
#         for speaker in os.listdir(os.path.join(single_input,data)):
#             command='sox '
#             for utt in os.listdir(os.path.join(single_input,data,speaker)):
#                 command += '%s '%os.path.join(single_input,data,speaker,utt)
#             command+='%s.wav'%os.path.join(data_path,'single_joint_wav',data,speaker)
#             os.system(command)





# audio generate stft data(numpy)
# every audio is cut to 3 seconds 
def audio_to_numpy(audio_path=audio_norm_path,data_outpath=database_path, fix_sr=sample_rate,data_classes=data_classes):
    print('start generate stft audio data...')
    
    for data in data_classes:
        mkdir(os.path.join(data_outpath,'single',data))
        for speaker in os.listdir(os.path.join(audio_path,data)):
            for utt in os.listdir(os.path.join(audio_path,data,speaker)):
                audio_data,_ = librosa.load(os.path.join(audio_path,data,speaker,utt),fix_sr)

                new_audio_data = ''
                if(len(audio_data) >= 3*fix_sr):
                    mkdir(os.path.join(data_outpath,'single',data,speaker))
                    new_audio_data = audio_data[0:3*fix_sr] #cut 0-3 s to new audio 
                    new_audio_data = utils.fast_stft(new_audio_data) #fast stft
                    
                    name = 'stft_'+os.path.splitext(utt)[0].split('_')[1]+'.npy'
                    np.save((os.path.join(data_outpath,'single',data,speaker,name)),new_audio_data)

                    with open('%s/single/%s_single_TF.txt' % (data_outpath,data), 'a') as f:
                        f.write('%s' % os.path.join(data_outpath,'single',data,speaker,name))
                        f.write('\n')
                    break
                else:
                    continue 
        
                



# Divided into n parts according to the number of speakers
def split_to_mix(data_path=database_path, partition=2,data_classes=data_classes):
    for data in data_classes:
        speakers = os.listdir(os.path.join(data_path,'single',data))
        part_len = len(speakers)//partition
        
        start = 0
        part_idx = 0
        split_list = []

        while((start + part_len) <= len(speakers)):
            part = speakers[start:(start+part_len)]

            for speaker in part:
                for utt in os.listdir(os.path.join(data_path,'single',data,speaker)):
                    path = os.path.join(data_path,'single',data,speaker,utt)
                    with open('%s/%s_single_TF_part%d.txt'%(data_path,data,part_idx),'a') as f:
                        f.write(path)
                        f.write('\n')
            start +=part_len 
            part_idx+=1



# Mix a single audio （numpy）
def single_mix(data,combo_idx, split_list, datapath):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = 'mix'
    mid_name = ''
    for part_idx in range(len(split_list)):
        stftdata_path = split_list[part_idx][combo_idx[part_idx]]
        stft_list = stftdata_path.split('/')
        data = stft_list[-3]
        speaker = stft_list[-2]
        stft_utt = os.path.splitext(stft_list[-1])[0].split('_')[-1]
        wav_path = os.path.join(audio_norm_path,data,speaker,'norm_%s.wav'%stft_utt)

        wav, _ = librosa.load(wav_path, sr=sample_rate)
        wav_list.append(wav[0:3*_]) #cut 0-3 s
        mid_name += '-%s' % speaker

    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav += wav * mix_rate #mix two signals

    wav_name = prefix + mid_name + '.wav'
    if not os.path.exists('%s/mix_wav/%s' % (datapath, data)):
        os.mkdir('%s/mix_wav/%s' % (datapath, data))
    wavfile.write('%s/mix_wav/%s/%s' % (datapath, data,wav_name), sample_rate, mix_wav)

    F_mix = utils.fast_stft(mix_wav)
    name = prefix + mid_name + '.npy'
    if not os.path.exists('%s/mix/%s' % (datapath, data)):
        os.mkdir('%s/mix/%s' % (datapath, data))
    store_path = '%s/mix/%s/%s' % (datapath, data,name)

    np.save(store_path, F_mix)

    with open('%s/%s_mix_log.txt' % (datapath,data), 'a') as f:
        f.write(name)
        f.write('\n')


# Mix all the audio to get n2 audio
def all_mix(data_path=database_path, partition=2,data_classes=data_classes):

    print('mixing data....')

    for data in data_classes:
        num_mix = 1
        num_mix_check = 0
        split_list= []
        for idx in range(partition):
            part = []
            f = open('%s/%s_single_TF_part%s.txt'%(data_path,data,idx))
            line = f.readline()
            while line:
                part.append(line)
                line = f.readline()                
            f.close()    
            split_list.append(part)

        part_len = len(split_list[-1])
        idx_list = [i for i in range(part_len)]

        idx_slice = random.sample(idx_list,15)  #about 30 hours data in total 
        num_mix  = part_len * len(idx_slice)
        print('number of mix data: ', num_mix)

        combo_idx_list = itertools.product(idx_list, idx_slice)  #2 speakers 
        for combo_idx in combo_idx_list:
            num_mix_check += 1
            single_mix(data,combo_idx, split_list, data_path)
            print('\rnum of completed mixing audio : %d' % num_mix_check, end='')
        print()


# Single audio generation complex mask map
def single_crm(data,idx_str_list, mix_path, data_path):
    F_mix = np.load(mix_path)
    mix_name = 'mix'
    mid_name = ''
    dataset_line = ''
    for idx in idx_str_list:
        mid_name += '-%s' % idx
        mix_name += '-%s' % idx
    mix_name += '.npy'
    dataset_line += mix_name

    for idx in idx_str_list:
        speaker_path = os.path.join(data_path,'single',data,idx)
        for utt in os.listdir(speaker_path):
            path = os.path.join(speaker_path,utt)
        # single_name = 'single-%s.npy' % idx
        # path = '%s/single/%s' % (data_path, single_name)
        F_single = np.load(path)
        cRM = utils.fast_cRM(F_single, F_mix)

        last_name = '-%s' % idx
        cRM_name = 'crm' + mid_name + last_name + '.npy'

        store_path = '%s/crm/%s/%s' % (data_path, data,cRM_name)
        if not os.path.exists('%s/crm/%s' % (data_path, data)):
            os.mkdir('%s/crm/%s' % (data_path, data))
        np.save(store_path, cRM)

        with open('%s/crm_log.txt' % data_path, 'a') as f:
            f.write(cRM_name)
            f.write('\n')

        dataset_line += (' ' + cRM_name)

    with open('%s/%s_dataset.txt' % (data_path,data), 'a') as f:
        f.write(dataset_line)
        f.write('\n')


# all audio generation complex mask map
def all_crm(data_classes = data_classes, data_path=database_path):
    for data in data_classes:
        mix_list =[]
        with open('%s/%s_mix_log.txt'%(data_path,data), 'r') as f:
            mix_list = f.read().splitlines()  

        for mix in mix_list:
            mix_path = '%s/mix/%s/%s' % (data_path, data,mix)
            mix = mix.replace('.npy', '')
            mix = mix.replace('mix-', '')
            idx_str_lsit = mix.split('-')
            single_crm(data,idx_str_lsit, mix_path, data_path)




if __name__ == '__main__':
    # init_dir()

    # audio_to_numpy(audio_norm_path,database_path,sample_rate,data_classes)


    # split_to_mix(database_path,num_speakers,data_classes)

    all_mix(database_path,num_speakers,data_classes)

    all_crm(data_classes,database_path)
