# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 08:29:06 2017

A single largest lung image was created by lung processor, a CNN then is run to extract additional features from the processed image

@author: linhaili
"""

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge, UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize

import os

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512
batch_size = 1

smooth = 1.
def load_images(path, patients, channel=0):    
    images = list() 
    lungSize = []
    for fn in patients:
        filename = os.path.join(path, fn+'.jpg')
        im = load_img(filename)
        im = img_to_array(im)
        im = im[channel,:,:]
        lungSize.append(im.shape[1]*im.shape[0])
        #normalize the image
        mean = np.mean(im[im>0])  
        std = np.std(im[im>0])
        old_min = np.min(im)   # background color
        im[im==old_min] = mean-1.2*std   # resetting backgound color
        im = im-mean
        im = im/std
        mean = np.mean(im)
        im = im - mean
        minP = np.min(im)
        maxP = np.max(im)
        im = im/(maxP-minP)
        im = resize(im, [img_rows, img_cols])
        im = np.expand_dims(im, axis=0) 
        #im = imread(filename)
        images.append(im)
        
    X = np.asarray(images)
    
    return X, lungSize

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(weights='unet.hdf5'):
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, (3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(conv9)
    
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    model.load_weights(weights)

    return model