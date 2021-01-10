from mtcnn.mtcnn import MTCNN
import cv2
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
from numba import jit
import time 

def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")

@jit()
def face_detect(input_dir,output_dir,data_classes,detector):
    mkdir(output_dir)
    for data_class in data_classes:
        mkdir(os.path.join(output_dir,data_class))
        
        data_path=os.path.join(input_dir,data_class)#get speaker dir path 

        
        for speaker in os.listdir(data_path):
            start = time.time()
            utt_path= os.path.join(data_path,speaker)#get utterance dir path 


            mkdir(os.path.join(output_dir,data_class,speaker))
            for utt in os.listdir(utt_path):
                frame_path = os.path.join(utt_path,utt)
                img = cv2.imread(frame_path)
                if type(img) is not np.ndarray:  #some images don't have any data
                    face_img=np.zeros((160,160,3))
                else:
                    face = detector.detect_faces(img)
                    if(len(face) == 0):
                        print(frame_path,' is no face!')
                        face_img = cv2.resize(img,(160,160))
                    else:
                        x,y,w,h = face[0]['box']
                        if(x<0):
                            x = 0
                        if(y<0):
                            y = 0
                        face_img = img[y:y+h,x:x+w,:]
                        face_img = cv2.resize(face_img,(160,160))
                cv2.imwrite('%s/%s/%s/%s'%(output_dir,data_class,speaker,utt),face_img)
            end = time.time()
            print(end-start)
            break
        break
    



        print('data class:',data_class,' Finished!!')
        
    print('ALL DATA HAS BEEN FINISHED!!!')


def test(detector):
    for image in os.listdir('/Work19/2020/lijunjie/LRS3/frames/pretrain/1X7fZoDs9KU/'):
        img = cv2.imread(os.path.join('/Work19/2020/lijunjie/LRS3/frames/pretrain/1X7fZoDs9KU/',image))
        print(image)
        face = detector.detect_faces(img)


if __name__ =='__main__':
    detector = MTCNN()
    # test(detector)
    input_frame_path = '/Work19/2020/lijunjie/LRS3/frames'
    output_face_path = './faces'
    data_classes = ['pretrain','trainval','test']
    face_detect(input_frame_path,output_face_path,data_classes,detector )




    # def bounding_box_check(faces,x,y):
#     # check the center
#     for face in faces:
#         bounding_box = face['box']
#         if(bounding_box[1]<0):
#             bounding_box[1] = 0
#         if(bounding_box[0]<0):
#             bounding_box[0] = 0
#         if(bounding_box[0]-50>x or bounding_box[0]+bounding_box[2]+50<x):
#             print('change person from')
#             print(bounding_box)
#             print('to')
#             continue
#         if (bounding_box[1]-50 > y or bounding_box[1] + bounding_box[3]+50 < y):
#             print('change person from')
#             print(bounding_box)
#             print('to')
#             continue
#         return bounding_box

# def face_detect(file,detector,frame_path,cat_train,output_dir):
#     name = file.replace('.jpg', '').split('-')
#     log = cat_train.iloc[int(name[0])]
#     x = log[3]
#     y = log[4]

#     img = cv2.imread('%s%s'%(frame_path,file))
#     x = img.shape[1] * x
#     y = img.shape[0] * y

#     faces = detector.detect_faces(img)
#     # check if detected faces
#     if(len(faces)==0):
#         print('no face detect: '+file)
#         return #no face
#     bounding_box = bounding_box_check(faces,x,y)
#     if(bounding_box == None):
#         print('face is not related to given coord: '+file)
#         return
#     print(file," ",bounding_box)
#     print(file," ",x, y)
#     crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
#     crop_img = cv2.resize(crop_img,(160,160))
#     cv2.imwrite('%s/frame_'%output_dir + name[0] + '_' + name[1] + '.jpg', crop_img)
#     #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
#     #plt.imshow(crop_img)
#     #plt.show()
