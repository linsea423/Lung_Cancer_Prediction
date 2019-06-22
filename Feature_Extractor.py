# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:45:26 2017

Extract features from the images

@author: linhaili
"""

import pandas as pd
import numpy as np
import time
import os

################ResNet50 Model############################
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

K.set_image_dim_ordering('tf')  # TensorFlow dimension ordering in this code

img_rows = 224
img_cols = 224
channels = 3
batch_size = 1

def load_pat_images(path, patients):    
    images = np.zeros((len(patients),img_rows,img_cols,channels))
    k = 0
    for fn in patients:
        filename = os.path.join(path, fn+'.jpg')
        im = load_img(filename, target_size=(img_rows, img_cols))
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        images[k] = im
        k += 1
    
    return images

resnetModel = ResNet50(include_top=False)
###########################################################

from New_Lung_Processor import *

#Feature columns
columns = ['numNode', 'totalVol', 'maxVol', 'minVol', 'maxHU', 'stdHU', 'meanHU', 'minHU', 'minCirc', \
       'maxEqDia', 'stdEqDia', 'maxArea', 'avgArea', 'totalArea', 'maxDia', 'dnodeX', 'dnodeY', 'FracConc', \
       'Acutance', 'Entropy', 'avgLungHU', 'maxLungHU', 'minLungHU', 'stdLungHU', 'bloodAreaFrac', 'fatAreaFrac', \
       'waterAreaFrac', 'tissueAreaFrac', 'largestLung'] 

#############################################################################
#start time
start = time.time()

# Load patients
dicom_root = 'E:\\Kaggle\\Data_Science_Bowl_2017\\stage1\\'

#get the list of patients in the training dataset from the label file
train_feature = pd.read_csv('stage1_labels.csv')
patients = train_feature['id']   

features = process_patient(patients, dicom_root, imgFolder='train')
for kc in range(len(columns)):
    train_feature[columns[kc]] = features[:,kc]
    
train_feature.to_csv('train_features.csv', index=False, float_format='%.4f')

train_feature = pd.read_csv('train_features.csv')
images = load_pat_images('train', patients)
print('Get CNN features from resnet50 for training data...')
cnn_features = resnetModel.predict(images, batch_size=batch_size)
cnn_features = cnn_features.reshape((len(patients),2048))
print('CNN features getted!')

#append the new features to the existing feature dataframe
for kf in range(cnn_features.shape[1]):
    train_feature['cnn_fea'+str(kf+1)] = cnn_features[:,kf]
             
train_feature.to_csv('train_features.csv', index=False, float_format='%.4f')
       
end = time.time()
print('Train Features: Used {:.2f} seconds'.format(end-start))

#########TEST DATA SET########################################################
start = time.time()

# Load patients
dicom_root = 'E:\\Kaggle\\Data_Science_Bowl_2017\\stage1\\'

#get the list of patients in the training dataset from the label file
test_feature = pd.read_csv('stage1_solution.csv')
patients = test_feature['id']   

features = process_patient(patients, dicom_root, imgFolder='test1')
for kc in range(len(columns)):
    test_feature[columns[kc]] = features[:,kc]
    
test_feature.to_csv('test_features.csv', index=False, float_format='%.4f')

test_feature = pd.read_csv('test_features.csv')    
print('Get features for testing dataset using trained ResNet50 Model...')
images = load_pat_images('test1', patients)
cnn_features = resnetModel.predict(images, batch_size=batch_size)
cnn_features = cnn_features.reshape((len(patients),2048))
print('Test dataset CNN features obtained!')

#append the new features to the existing feature dataframe
for kf in range(cnn_features.shape[1]):
    test_feature['cnn_fea'+str(kf+1)] = cnn_features[:,kf]
      
test_feature.to_csv('test_features.csv', index=False, float_format='%.4f')
        
end = time.time()
print('Test Features: Used {:.2f} seconds'.format(end-start))

#########STAGE 2 TEST DATA SET#################################################
start = time.time()

# Load patients
dicom_root = 'E:\\Kaggle\\Data_Science_Bowl_2017\\stage2\\'

#get the list of patients in the training dataset from the label file
test_feature2 = pd.read_csv('stage2_sample_submission.csv')
patients = test_feature2['id']   

features = process_patient(patients, dicom_root, imgFolder='test2')
for kc in range(len(columns)):
    test_feature2[columns[kc]] = features[:,kc]
    
test_feature2.to_csv('stage2_features.csv', index=False, float_format='%.4f')

test_feature2 = pd.read_csv('stage2_features.csv')    
print('Get features for testing dataset using trained ResNet50 Model...')
images = load_pat_images('test2', patients)
cnn_features = resnetModel.predict(images, batch_size=batch_size)
cnn_features = cnn_features.reshape((len(patients),2048))
print('Test dataset CNN features obtained!')

#append the new features to the existing feature dataframe
for kf in range(cnn_features.shape[1]):
    test_feature2['cnn_fea'+str(kf+1)] = cnn_features[:,kf]
      
test_feature2.to_csv('stage2_features.csv', index=False, float_format='%.4f')
        
end = time.time()
print('Test Features: Used {:.2f} seconds'.format(end-start))