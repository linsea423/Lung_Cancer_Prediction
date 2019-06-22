# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:34:06 2017

USE UNET to extract features from the single composed lung segments

@author: linhaili
"""

from Lung_CNN_Module import *

import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

#Calculate the fraction of concavity
def frac_concave(cnt):
    cntLength = cv2.arcLength(cnt, True)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    frac = 0
    
    if len(hull) < len(cnt): #The contour is NOT convex
        defects = cv2.convexityDefects(cnt,hull)   
    
        #sort the defects
        defects = defects[np.argsort(defects[:,:,0].flatten())]    
        
        s1,e1,_,_ = defects[0,0]
        darc = cv2.arcLength(cnt[s1:e1], False)           
        for i in range(1, defects.shape[0]):
            s,e,_,_ = defects[i,0]     
            
            if s<e1:
                s = e1            
                
            darc += cv2.arcLength(cnt[s:e], False)
            
            s1,e1 = s, e
        
        frac = darc/cntLength
    
    return frac

#shrink or enlarge a contours
def transform_contour(cnt, scale=0.2):
    
#    mask = np.zeros(img.shape, dtype=np.uint8)
#    mask = cv2.fillPoly(mask, [cnt], 255)
#    img[mask==0] = 0
#    M = cv2.moments(img)
    M = cv2.moments(cnt)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']
    
    shrCnt = np.empty((len(cnt),1,2), dtype=np.int)
    enlCnt = np.empty((len(cnt),1,2), dtype=np.int)
    for i in range(len(cnt)):
        cntX = cnt[i,0,0]
        cntY = cnt[i,0,1]
        if cntX<cx:
            shrCnt[i,0,0] = int(cntX + abs(cx - cntX)*scale)
            enlCnt[i,0,0] = int(cntX - abs(cx - cntX)*scale)
        else:
            shrCnt[i,0,0] = int(cntX - abs(cx - cntX)*scale)
            enlCnt[i,0,0] = int(cntX + abs(cx - cntX)*scale)
        
        if cntY<cy:
            shrCnt[i,0,1] = int(cntY + abs(cy - cntY)*scale)
            enlCnt[i,0,1] = int(cntY - abs(cy - cntY)*scale)
        else:
            shrCnt[i,0,1] = int(cntY - abs(cy - cntY)*scale)
            enlCnt[i,0,1] = int(cntY + abs(cy - cntY)*scale)          
        
    return shrCnt, enlCnt      

#calculate the boundary gradient
#modified from http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=887618
def boundary_gradient(img, cnt, scale=0.25):

    shrCnt, enlCnt = transform_contour(cnt, scale=scale) #Get the smaller and larger contour
    cntK, _ = transform_contour(cnt, scale=0.025) #shrink the contour a little bit in order to avoid the lung area
    
    pmax = -2000
    pmin = 2000
    di  = 0 
    for kP in range(len(cntK)):
        lnMask = np.zeros(img.shape, np.uint8)
        lnMask = cv2.line(lnMask, tuple(cntK[kP][0]), tuple(shrCnt[kP][0]), 255, 1)
        ni = np.sum(lnMask==255)
        lPix = img[lnMask==255].flatten()
        di += np.sqrt(np.sum(np.diff(lPix)**2)/ni)
        if np.max(lPix)>pmax:
            pmax = np.max(lPix)
        if np.min(lPix)<pmin:
            pmin = np.min(lPix)
            
    Adg = di/(pmax-pmin)/len(cntK) #accutance    
    
    return Adg

#calculate the entropy of the node
def cal_entropy(img, cnt):
    img = np.float32(img)
    mask = np.zeros(img.shape, np.uint8)
    mask = cv2.fillPoly(mask, [cnt], 255)
    histPix = cv2.calcHist([img], [0], mask, [100], [-1, 1])
    
    histPix = histPix.ravel()/histPix.sum()
    logs = np.log2(histPix+0.00001)
    entropy = -1 * (histPix*logs).sum()
    
    return entropy  


#Feature columns
columns = ['numNode', 'maxHU', 'stdHU', 'meanHU', 'minHU', 'minCirc', \
       'maxEqDia', 'stdEqDia', 'maxArea', 'avgArea', 'totalArea', 'maxDia', 'dnodeX', 'dnodeY', 'FracConc', \
       'Acutance', 'Entropy'] 

def extract_unet_features(images, masks, lungSize):
    featureMat = np.empty((0,17))
    
    for km in range(len(masks)):
        nodeMask = masks[km,0,:,:]
        nodeMask[nodeMask>0.5] = 255
        nodeMask[nodeMask<=0.5] = 0
        nodeMask = np.uint8(nodeMask)
        _, contours, _ = cv2.findContours(nodeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#        plt.figure(figsize=(12,6))
#        fMask = np.zeros(nodeMask.shape, np.uint8)
#        fMask = cv2.fillPoly(fMask, contours, 255)
#        plt.subplot(121)
#        plt.imshow(fMask, cmap='gray')
        
        if len(contours)<1:
            #features for those without a node
            features = np.asarray([0,-1,1,-1,-1,1,0,0,0,0,0,0,256,512,0,1,20])
            featureMat = np.vstack((featureMat, features))
            continue
        
        for kc in range(len(contours)-1,-1,-1):
            #Remove very small objects and V-shape objects
            _,radius = cv2.minEnclosingCircle(contours[kc])
            rect = cv2.minAreaRect(contours[kc])
            box = cv2.boxPoints(rect)
            w1 = np.sqrt(np.sum((box[0]-box[1])**2))
            w2 = np.sqrt(np.sum((box[1]-box[2])**2))
            width = max(w1, w2)
            height = min(w1,w2)
            boxArea = width*height
            cntArea = cv2.contourArea(contours[kc])
            cntLength = cv2.arcLength(contours[kc], True)
            if radius<4 or width<3 or height<3: #if too small
                contours.pop(kc)
                continue
            if cntArea<boxArea/2: #v-shape objects                
                contours.pop(kc)
                continue
            #The nodes cannot be larger than 10% of the image area
            if cntArea>0.10*nodeMask.shape[0]*nodeMask.shape[1]:
                contours.pop(kc)
                continue
            #Remove very long objects
            if width/height>3:                
                contours.pop(kc)
                continue
            if width>40:
                contours.pop(kc)
                continue
            
            #if the nodes are in the middle of the window, probly tissues
            M = cv2.moments(contours[kc])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if (abs(cx-256)<100) and (abs(cy-256)<75):
                contours.pop(kc)
                continue
            
            #Further remove the candidate node base on shape features
            Adg = boundary_gradient(images[km,0], contours[kc], scale=0.3)
            #the threshold is about to find    
            if Adg>0.4:
                contours.pop(kc)
                continue 
            
            #Remove the node that are not very uniform
            En = cal_entropy(images[km,0], contours[kc])
            #the threshold is about to find    
            if En>4:
                contours.pop(kc)
                continue 
        
#        plt.subplot(122)
#        fMask = np.zeros(nodeMask.shape, np.uint8)
#        fMask = cv2.fillPoly(fMask, contours, 255)
#        plt.imshow(fMask, cmap='gray')
#        plt.show()
        
        if len(contours)<1:
            #features for those without a node
            features = np.asarray([0,-1,1,-1,-1,1,0,0,0,0,0,0,256,512,0,1,20])
            featureMat = np.vstack((featureMat, features))
            continue   
        
        numNode = len(contours)
        
        nodeAtt = np.empty((0,))
        nodeArea = []
        nodeEqDia = []
        maxDia = 0 
        maxfracConc = 0
        minAdg = 1000
        minEn = 1000
        minCirc = 1000
        
        tnArea = 0
        
        for cnt in contours:        
           mask = np.zeros((512,512), np.uint8)
           mask = cv2.fillPoly(mask, [cnt], 255)
           #Some features about the shape of the node
           cntArea = cv2.contourArea(cnt)
           nodeArea.append(cntArea)
           arcLeng = cv2.arcLength(cnt, True)
           circ = 4*np.pi*cntArea/arcLeng**2
           nodeEqDia.append(2*np.sqrt(cntArea/np.pi))
           
           #Get the maximum diameter of the node
           _,_,w,h = cv2.boundingRect(cnt)
           cmaxDia = max(w,h)
           #Some features of attenuation for the node
           nodeAtt = np.hstack((nodeAtt, images[km,0][mask!=0].flatten()))
           
           #Get the fraction of concavity of the dominant node
           fracConc = frac_concave(cnt)
           #get the boundary gradient
           Adg = boundary_gradient(images[km,0], cnt, scale=0.3)
           #get the node entropy
           En = cal_entropy(images[km,0], cnt)
            
           #Obtain the location of the largest node
           #If there are still nodes within the window 75 pixels away from the lung center, remove them
           M = cv2.moments(cnt)
           cx = int(M['m10']/M['m00'])
           cy = int(M['m01']/M['m00'])
           if cntArea>tnArea:
               tnArea = cntArea
               dnodeX = cx
               dnodeY = cy  
                
           if minCirc>circ:
                minCirc = circ
            
           if maxDia<cmaxDia:
                maxDia = cmaxDia
                
           if maxfracConc<fracConc:
                maxfracConc = fracConc
            
           if minAdg>Adg:
                minAdg = Adg
            
           if minEn>En:
                minEn = En
    
        maxEqDia = np.max(nodeEqDia)
        stdEqDia = np.std(nodeEqDia)
        maxArea = np.max(nodeArea)/lungSize[km]
        avgArea = np.mean(nodeArea)/lungSize[km]
        totalArea = np.sum(nodeArea)/lungSize[km]
        #Features about attenuation of the node
        maxAtt = np.max(nodeAtt)
        stdAtt = np.std(nodeAtt)
        minAtt = np.min(nodeAtt)  
        meanAtt = np.mean(nodeAtt)
        #Compile all the features together
        features = np.asarray([numNode, maxAtt, stdAtt, meanAtt, \
                               minAtt, minCirc, maxEqDia, stdEqDia, maxArea, avgArea, \
                               totalArea, maxDia, dnodeX, dnodeY, maxfracConc, minAdg, minEn])
        featureMat = np.vstack((featureMat, features))
        
    return featureMat

#get the unet model
model = get_unet(weights='unet.hdf5')

##################################################################################
#Get the data    
train_data = pd.read_csv('train_features.csv')
patients = train_data['id']
print('#################Get NODE MASK for Training data....')
images,lungSize = load_images('train', patients, channel=1)
masks = model.predict(images, batch_size=1)

#Get the features of the node
unetFeature = extract_unet_features(images, masks, lungSize)

#update the feature matrix to the training dataset
train_data[columns] = unetFeature
train_data.to_csv('train_features_unet.csv', index=False, float_format='%.4f')
print('#################Node features exacted for Training data....')

##################################################################################          
#Do the same with the test datasets
#Get the data    
test_data = pd.read_csv('test_features.csv')
patients = test_data['id']
print('#################Get NODE MASK for Testing data....')
images, lungSize = load_images('test1', patients, channel=1)
masks = model.predict(images, batch_size=1)
#Get the features of the node
unetFeature = extract_unet_features(images, masks, lungSize)
test_data[columns] = unetFeature
         
test_data.to_csv('test_features_unet.csv', index=False, float_format='%.4f')
print('#################Node features exacted for Testing data....')

##################################################################################
#Do the same with the stage 2 test datasets
#Get the data    
test_data2 = pd.read_csv('stage2_features.csv')
patients = test_data2['id']
print('#################Get NODE MASK for Stage 2 Testing data....')
images, lungSize = load_images('test2', patients, channel=1)
masks = model.predict(images, batch_size=1)
#Get the features of the node
unetFeature = extract_unet_features(images, masks, lungSize)
test_data2[columns] = unetFeature
         
test_data2.to_csv('stage2_features_unet.csv', index=False, float_format='%.4f')
print('#################Node features exacted for Stage 2 Testing data....')