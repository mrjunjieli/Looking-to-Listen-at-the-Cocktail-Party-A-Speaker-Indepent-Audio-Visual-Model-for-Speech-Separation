#create:20/11
#last editor:Junjie Li

import tensorflow as tf
import sys
sys.path.append('../utils/')
import utils
from tensorflow.python.framework import tensor_util
import numpy as np
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

## paraeter
MODEL_PATH = '../FaceNetmodel/keras/facenet_keras.h5'
# VALID_FRAME_LOG_PATH = '../../data/video_data/valid_face_text.txt'
FACE_INPUT_PATH = '/Work19/2020/lijunjie/LRS3/faces/'

audio_input_path='/Work19/2020/lijunjie/LRS3/AV_model_database/single'

save_path = '/Work19/2020/lijunjie/LRS3/face1022_emb/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

data_classes = ['pretrain','trainval','test']



###############
model = load_model(MODEL_PATH)
model.summary()
avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)

for data_class in data_classes:
    for speaker in os.listdir(os.path.join(FACE_INPUT_PATH,data_class)):
        audio_speaker_path = os.path.join(audio_input_path,data_class,speaker)
        name=''
        embtmp = np.zeros((75,1,1792))
        for utt in os.listdir(audio_speaker_path):
            name = os.path.splitext(utt)[0].split('_')[1]
        for i in range(1,76):
            face_name = '%05d-%04d.jpg'%(int(name),i)
            I = mpimg.imread(os.path.join(FACE_INPUT_PATH,data_class,speaker,face_name))
            I_np = np.array(I)
            I_np = I_np[np.newaxis, :, :, :]
            embtmp[i - 1, :] = avgPool_layer_model.predict(I_np)
        if not os.path.exists(os.path.join(save_path,data_class)):
            os.makedirs(os.path.join(save_path,data_class))
        np.save(os.path.join(save_path,data_class,speaker+'_faceemb.npy'),embtmp)
        with open(os.path.join(save_path,"%s_faceemb.txt"%data_class),'a') as f:
            f.write(os.path.join(save_path,data_class,speaker+'_faceemb.npy\n'))