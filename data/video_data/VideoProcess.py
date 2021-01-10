#extract frames from videodata set  fps=25
#out dataset is LRS3 
#author JunjieLi
#create 2020/11


import os
import datetime
import pandas as pd
import time


def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")


def extract_images_from_videos(input_dir,output_dir,data_classes):
    mkdir(output_dir)
    for data_class in data_classes:
        mkdir(os.path.join(output_dir,data_class))
        data_path=os.path.join(input_dir,data_class)#get speaker dir path 
        for speaker in os.listdir(data_path):
            command = ''
            utt_path= os.path.join(data_path,speaker)#get utterance dir path 
            mkdir(os.path.join(output_dir,data_class)+'/'+speaker)
            for utt in os.listdir(utt_path):
                if(os.path.splitext(utt)[1]=='.mp4'):                    
                    command += 'ffmpeg -i %s -vf fps=25 -f image2 %s/%s-%%04d.jpg;'% (os.path.join(utt_path,utt), os.path.join(output_dir+data_class,speaker), os.path.splitext(utt)[0])            
            os.system(command)
            print('speaker:',speaker,' in ',data_class,' has been processed!')
            print('---------------------------')
        
        print('data class:',data_class,' Finished!!')
        
    print('ALL DATA HAS BEEN FINISHED!!!')
    


if __name__ == "__main__":
    input_dir = "/Work19/2020/lijunjie/LRS3"
    output_dir = "/Work19/2020/lijunjie/LRS3/frames"
    data_classes=['pretrain','trainval','test']
    extract_images_from_videos(input_dir,output_dir,data_classes)


