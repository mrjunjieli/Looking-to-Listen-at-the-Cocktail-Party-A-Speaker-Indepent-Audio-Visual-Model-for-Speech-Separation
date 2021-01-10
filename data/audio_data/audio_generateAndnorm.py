#author :Junjie Li 
#out dataset is LRS3 
#create time:2020/11


import os
import pandas as pd
import time
import librosa 
import numpy as np 
import scipy.io.wavfile as wavfile





def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")


def extract_audio(input_dir,output_dir,data_classes):
    #extract audio from videodata set 
    mkdir(output_dir)
    for data_class in data_classes:
        mkdir(os.path.join(output_dir,data_class))
        data_path=os.path.join(input_dir,data_class)#get speaker dir path 
        for speaker in os.listdir(data_path):
            command = ''
            utt_path= os.path.join(data_path,speaker)#get utterance dir path 
            mkdir(os.path.join(output_dir,data_class,speaker))
            for utt in os.listdir(utt_path):
                if(os.path.splitext(utt)[1]=='.mp4'):
                    
                    command += 'ffmpeg -i %s %s.wav;'% (os.path.join(utt_path,utt), os.path.join(output_dir,data_class,speaker,os.path.splitext(utt)[0]))            
            os.system(command)
            print('speaker:',speaker,' in ',data_class,' has been processed!')
            print('---------------------------')
        
        print('data class:',data_class,' Finished!!')
        
    print('ALL DATA HAS BEEN FINISHED!!!')
    

def audio_norm(input_dir,output_dir,data_classes,sample_rate):
    #normalization 
    mkdir(output_dir)
    for data_class in data_classes:
        mkdir(os.path.join(output_dir,data_class))
        data_path=os.path.join(input_dir,data_class)#get speaker dir path 
        for speaker in os.listdir(data_path):
            utt_path= os.path.join(data_path,speaker)#get utterance dir path 
            mkdir(os.path.join(output_dir,data_class,speaker))
            for utt in os.listdir(utt_path):
                if(os.path.splitext(utt)[1]=='.wav'):
                    audio,_ = librosa.load(os.path.join(utt_path,utt),sample_rate)
                    max = np.max(np.abs(audio))
                    norm_audio = np.divide(audio,max)
                    wavfile.write(os.path.join(output_dir,data_class,speaker,'norm_'+utt),sample_rate,norm_audio)
            print('speaker:',speaker,' in ',data_class,' has been processed!')
            print('---------------------------')
        
        print('data class:',data_class,' Finished!!')
        
    print('ALL DATA HAS BEEN FINISHED!!!')


if __name__ == "__main__":
    input_dir = "/Work19/2020/lijunjie/LRS3"
    output_dir = "/Work19/2020/lijunjie/LRS3/audio"


    data_classes=['pretrain','trainval','test']
    extract_audio(input_dir,output_dir,data_classes)
    
    input_norm=output_dir
    output_norm='/Work19/2020/lijunjie/LRS3/audio_norm'
    sample_rate = 16000
    audio_norm(input_norm,output_norm,data_classes,sample_rate)
