# -*- coding: utf-8 -*-
"""
Created on Sun Apr 2 13:43:00 2017

Newly created to process the images to extract features and form a single composited lung for each patient

@author: linhaili
"""

# Standard imports
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

from scipy.misc import imsave, imread

from Lung_CNN_Module import *
#UNET model
unetmodel = get_unet(weights='unet.hdf5')

# Pandas configuration
#pd.set_option('display.max_columns', None)
#print('OK.')
#print(cv2.__version__)

# get patients list
import dicom

# DICOM rescale correction
def rescale_correction(s):
    s.image = s.pixel_array * s.RescaleSlope + s.RescaleIntercept

# Returns a list of images for that patient_id, in ascending order of Slice Location
# The pre-processed images are stored in ".image" attribute
def load_patient(patient_id, dicom_root):
    files = glob.glob(dicom_root + '{}\\*.dcm'.format(patient_id))
    slices = []
    for f in files:
        dcm = dicom.read_file(f)
        rescale_correction(dcm)
        slices.append(dcm)
    
    try:
        slices = sorted(slices, key=lambda x: x.SliceLocation)
    except:          
        slices = sorted(slices, key=lambda x: x.InstanceNumber)    
        
    return slices

#Calculate the angle of the fish relative to the horizon, which will be used to crop the fish from the image later 
def side_of_line(x1,y1,x2,y2,x,y):
    side = np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))
    return side

def fill_indentation(cnt, thresh=1.5):    
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    
    mask = np.ones(len(cnt), dtype=bool)          
               
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = cnt[s][0]
        end = cnt[e][0]
               
        de = np.sqrt(np.sum((start-end)**2))        
        darc = cv2.arcLength(cnt[s:e], False)
        
        #Get the farthest distance of the points on the arc to the straight line connected by start and end
        fdis = d/256.
        
        if (darc/de>thresh) and (de<=60) and (min(de,fdis)/max(de,fdis)<0.5):
            mask[s:e] = False

    newContour = cnt[mask]
    
    return newContour                         

#separate the lungs into left and right
def separate_lungs(contour,img):
    x,y,w,h = cv2.boundingRect(contour[0])
    
    if x<img.shape[1]/2 and w>img.shape[1]/2.5:            
        m = int(x + w/2)
        #n = int(y + h/2)
        numWhite = []
        allIdx = list(range(m-30,m+31))
        for mm in allIdx:
            temp = np.sum((img[:,mm] !=0 )*1.)
            numWhite.append(temp)
        idx = np.argsort(np.array(numWhite))[0:2]
        idx = np.sort(idx)
        leftx = allIdx[idx[0]]
        lefty = np.where(img[:,allIdx[idx[0]]]!=0)[0][-1]
        rightx = allIdx[idx[1]]
        righty = np.where(img[:,allIdx[idx[1]]]!=0)[0][0]
        cv2.line(img, (leftx,lefty), (rightx, righty), 0, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        #cv2.imshow('img', img)
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        #remove the very small contours
        for k in range(len(contours)-1, -1, -1):
            if len(contours[k])<4:
                contours.pop(k)
                continue
            area = cv2.contourArea(contours[k])
            _,_,w,h=cv2.boundingRect(contours[k])
            if (area < img.shape[0]*img.shape[1]/50) or (w<5) or (h<5):
                contours.pop(k)
    else:
        contours = contour
    
    return contours

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
    
    if M['m00']==0:
        shrCnt=[]
        enlCnt=[]
        return shrCnt, enlCnt
    
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
    
    if len(shrCnt)<1:
        Adg = 1
        return Adg
    
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
#    pixels = img[mask==255].flatten()
#    minPix = int(np.min(pixels))
#    maxPix = int(np.max(pixels))
#    numHist = int(maxPix - minPix)
    histPix = cv2.calcHist([img], [0], mask, [600], [-301, 300])
    
    histPix = histPix.ravel()/histPix.sum()
    logs = np.log2(histPix+0.00001)
    entropy = -1 * (histPix*logs).sum()
    
    return entropy

#deal with infarction and obsecis
#remove the node that has low possibility to be a tumor
def remove_infarction(lungImg):    
    #refer to the original image in order to update the node masks
    tempImg = lungImg.copy()
    tempImg[tempImg>-300] = 255
    tempImg[tempImg<-300] = 0
    tempImg = np.uint8(tempImg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_ERODE, kernel)
    tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_OPEN, kernel)
    tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_DILATE, kernel)
    _,contours,_ = cv2.findContours(tempImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
        return lungImg           
                                
    for k in range(len(contours)-1, -1, -1):
        if len(contours)<1:
            return lungImg
        
        #Remove very small objects and V-shape objects
        _,radius = cv2.minEnclosingCircle(contours[k])
        rect = cv2.minAreaRect(contours[k])
        box = cv2.boxPoints(rect)
        w1 = np.sqrt(np.sum((box[0]-box[1])**2))
        w2 = np.sqrt(np.sum((box[1]-box[2])**2))
        width = max(w1, w2)
        height = min(w1,w2)
        cntArea = cv2.contourArea(contours[k])
        cntLength = cv2.arcLength(contours[k], True)
        #The nodes cannot be larger than 10% of the image area
        if cntArea>0.10*lungImg.shape[0]*lungImg.shape[1]:
            tempMask = np.zeros(lungImg.shape, np.uint8)
            tempMask = cv2.fillPoly(tempMask, [contours[k]], 255)
            lungImg[tempMask != 0] = -2000
            continue
        #Remove the nodes that are very round and smooth
        if (cntArea/(np.pi*radius**2)>0.85) and (cntLength/(2*np.pi*radius)<1.25) and (width>40):
            tempMask = np.zeros(lungImg.shape, np.uint8)
            tempMask = cv2.fillPoly(tempMask, [contours[k]], 255)
            lungImg[tempMask != 0] = -2000
            continue
        #further remove the objects with high circularity, benign
        circ = 4*np.pi*cntArea/cntLength**2
        if circ>0.8:
            tempMask = np.zeros(lungImg.shape, np.uint8)
            tempMask = cv2.fillPoly(tempMask, [contours[k]], 255)
            lungImg[tempMask != 0] = -2000
            continue
          
        #Deal with obsces and infarction cases
        _,_,w,h = cv2.boundingRect(contours[k])
        width = max(w,h)
        cntArea = cv2.contourArea(contours[k])
        roundness= 4*cntArea/(np.pi*width**2)
        if (roundness>0.5 and width>45):
            tempMask = np.zeros(lungImg.shape, np.uint8)
            tempMask = cv2.fillPoly(tempMask, [contours[k]], 255)
            lungImg[tempMask != 0] = -2000
            contours.pop(k)
            continue  
    
    return lungImg

#remove the node that has low possibility to be a tumor
def remove_benign_nodes(contours, lungImg, lungBound, xSpace, ySpace):
    #coordinate of the center, leftmost, rightmost, topmost, and bottomost of the lung
    lungCx = lungBound[0]
    lungCy = lungBound[1]
    lungLeftx = lungBound[2]
    lungRightx = lungBound[3]
    lungTopy = lungBound[4]
    lungBottomy = lungBound[5]
    leftLimit = lungLeftx + (lungCx - lungLeftx)/1.8
    rightLimit = lungRightx - (lungRightx - lungCx )/1.8
    toplimit = lungTopy + (lungCy - lungTopy)/1.8
    bottomlimit = lungBottomy - (lungBottomy - lungCy)/1.8
    
    #refer to the original image in order to update the node masks
    newMask = np.zeros(lungImg.shape, np.uint8)
    newMask = cv2.fillPoly(newMask, contours, 255)
    tempImg = lungImg.copy()
    tempImg[newMask==0] = -2000
    tempImg[tempImg>=-100] = 255
    tempImg[tempImg<-100] = 0
    tempImg = np.uint8(tempImg)
    _,contours,_ = cv2.findContours(tempImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
        contours = []
        return contours, lungImg           
                                
    for k in range(len(contours)-1, -1, -1):
        if len(contours)<1:
            contours = []
            return contours, lungImg
        
        #Remove very small objects and V-shape objects
        _,radius = cv2.minEnclosingCircle(contours[k])
        rect = cv2.minAreaRect(contours[k])
        box = cv2.boxPoints(rect)
        w1 = np.sqrt(np.sum((box[0]-box[1])**2))
        w2 = np.sqrt(np.sum((box[1]-box[2])**2))
        width = max(w1, w2)
        height = min(w1,w2)
        boxArea = width*height
        cntArea = cv2.contourArea(contours[k])
        cntLength = cv2.arcLength(contours[k], True)
        if (radius*xSpace<=3) or (width*xSpace<=3) or (height*ySpace<=3): #small objects
            contours.pop(k)
            continue
        #The nodes cannot be larger than 10% of the image area
        if cntArea>0.10*lungImg.shape[0]*lungImg.shape[1]:
            contours.pop(k)
            continue
        #Remove very long objects
        if width/height>3:
            contours.pop(k)
            continue
        if width>45:
            contours.pop(k)
            continue            
        
        #Further remove the candidate node base on shape features
        Adg = boundary_gradient(lungImg, contours[k], scale=0.3)
        #the threshold is about to find    
        if Adg>0.4:
            contours.pop(k)
            continue 
        
        #Remove the node that are not very uniform
        En = cal_entropy(lungImg, contours[k])
        #the threshold is about to find    
        if En>5:
            contours.pop(k)
            continue 
        
        #If there are nodes within the center of lung, remove them because studies show it is unlikely to be the node
        M = cv2.moments(contours[k])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if (abs(cx-256)<100) and (abs(cy-256)<75):
            contours.pop(k)
            continue
        if len(contours)<1:
            continue   
    
    return contours, lungImg

#Define a function to extract nodes information
def extract_node_features(nodeMasks, pat, largestLung, connContours, lungBound, xspace, yspace, zspace):   
    #The two contours for the connected tissues in the middle of the image
    connCnt1 = connContours[0]
    connCnt2 = connContours[1]
    
    #coordinate of the center, leftmost, rightmost, topmost, and bottomost of the lung
    lungCx = lungBound[0]
    lungCy = lungBound[1]
    lungLeftx = lungBound[2]
    lungRightx = lungBound[3]
    lungTopy = lungBound[4]
    lungBottomy = lungBound[5]
    leftLimit = lungLeftx + (lungCx - lungLeftx)/2
    rightLimit = lungRightx - (lungRightx - lungCx )/2
    toplimit = lungTopy + (lungCy - lungTopy)/1.8
    bottomlimit = lungBottomy - (lungBottomy - lungCy)/1.8
    
    nodeMasks = np.asarray(nodeMasks)
    
    #Crease a super mask to include all candidate node masks on a single image
    superMask = np.sum(nodeMasks, axis=0).clip(0,255)
    superMask = np.uint8(superMask)
    
    _,contours,_ = cv2.findContours(superMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
        features = np.asarray([0,0,0,0,-500,-500,-500, \
                               -500, 1, 0, 0, 0, 0, \
                               0, 0, 256, 512, 0, 1, 10])
        return features
    
    numNode = 0
    fracConc = 0
    nodeVol = []
    nodeCirc = []
    nodeArea = []
    nodeEqDia = []
    maxDia = [] 
    En = []
    maxfracConc = 0
    minAdg = 1000
    
    nodeAtt = np.empty((0,))
    
    tnArea = 0 #temp parameter to record the largest area of current node
    
    for k in range(len(contours)-1, -1, -1):
        cntB = contours[k]
        xr,yr,w,h = cv2.boundingRect(cntB)
        x = int(xr + w/2)
        y = int(yr + h/2)
        
        idx = []
        for km in range(len(nodeMasks)):
            img = nodeMasks[km].copy()
            _,cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                xr,yr,w,h = cv2.boundingRect(cnt)
                xm = int(xr + w/2)
                ym = int(yr + h/2)
                if (abs(x-xm)<=5) and (abs(y-ym)<=5):
                    idx.append(km)
                    break
                    
        #xyArray = nodeMasks[:,y,x]    
        #idx = np.where(xyArray != 0)[0]
        
        #it is not considered to be a node if it does not appear on at least 2 slices
        if len(idx)<1:
            contours.pop(k)
            continue
        
        cnodeAtt = np.empty((0,))
        cnodeArea = []
        cnodeCirc = []
        cnodeEqDia = []
        fracConc =[]
        Adg = []
        cmaxDia = 0
        for idxC in idx:
           img = nodeMasks[idxC].copy()
           _,cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
           lungImg = pat[idxC].copy()
           for cnt in cnts:                
               xr,yr,w,h = cv2.boundingRect(cnt)
               xm = int(xr + w/2)
               ym = int(yr + h/2)
               if (abs(x-xm)<=5) and (abs(y-ym)<=5):
                   mask = np.zeros(img.shape, np.uint8)
                   mask = cv2.fillPoly(mask, [cnt], 255)
                   #Some features about the shape of the node
                   cntArea = cv2.contourArea(cnt)
                   arcLeng = cv2.arcLength(cnt, True)
                   circ = 4*np.pi*cntArea/arcLeng**2
                   eqDia = 2*np.sqrt(cntArea/np.pi)
                   cnodeArea.append(cntArea)
                   cnodeCirc.append(circ)
                   cnodeEqDia.append(eqDia)
                   #Get the maximum diameter of the node
                   _,_,w,h = cv2.boundingRect(cnt)
                   w = max(w,h)
                   cmaxDia = cmaxDia if cmaxDia>w else w
                   #Some features of attenuation for the node
                   cnodeAtt = np.hstack((cnodeAtt, lungImg[mask!=0].flatten()))
                   
                   #Get the fraction of concavity of the dominant node
                   fracConc.append(frac_concave(cnt))
                   #get the boundary gradient
                   Adg.append(boundary_gradient(lungImg, cnt, scale=0.3))
                   #get the node entropy
                   En.append(cal_entropy(lungImg, cnt))
                   break
               
        cnodeAtt1 = cnodeAtt[cnodeAtt>-10]
        if len(cnodeAtt1)==0:
            contours.pop(k)
            continue
        else:
            maxAtt = np.max(cnodeAtt1)
            minAtt = np.min(cnodeAtt1)
            stdAtt = np.std(cnodeAtt1)
            meanAtt = np.mean(cnodeAtt1)
            if maxAtt<10 or meanAtt<5:
                contours.pop(k)
                continue
        
        #Obtain the location of the largest node
        #If there are still nodes within the window 75 pixels away from the lung center, remove them
        M = cv2.moments(cntB)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if np.max(cnodeArea)>tnArea:
            tnArea = np.max(cnodeArea)
            dnodeX = cx/lungCx
            dnodeY = cy/lungCy
            
        #Get the max fraction of concavity
        if np.max(fracConc)>maxfracConc:
            maxfracConc = np.max(fracConc)
            
        #Get the min boundary gradient
        if np.min(Adg)<minAdg:
            minAdg = np.min(Adg)
               
        #Because the node should be more like a sphere, therefore the area at different slice should change, 
        #or else it might not be a node
        numNode += 1
        avgArea = np.mean(cnodeArea)
        if np.max(idx)==np.min(idx):
            nodeVol.append(zspace*avgArea*xspace*yspace)
        else:
            nodeVol.append((np.max(idx)-np.min(idx))*zspace*avgArea*xspace*yspace)
            
        nodeAtt = np.hstack((nodeAtt, cnodeAtt))
        nodeCirc.append(np.mean(cnodeCirc))
        nodeEqDia.append(np.max(cnodeEqDia))
        nodeArea.append(np.max(cnodeArea))
        if (np.max(idx)-np.min(idx))*zspace > cmaxDia*xspace:
            maxDia.append((np.max(idx)-np.min(idx))*zspace)
        else:
            maxDia.append(cmaxDia)
    
    if numNode>0 and numNode<3:        
        totalVol = np.sum(nodeVol)/largestLung
        maxVol = np.max(nodeVol)
        minVol = np.min(nodeVol)
        #Features about the shape
        minCirc = np.min(nodeCirc)
        maxEqDia = np.max(nodeEqDia)*xspace
        stdEqDia = np.std(nodeEqDia)*xspace
        maxArea = np.max(nodeArea)*xspace**2
        avgArea = np.mean(nodeArea)*xspace**2
        totalArea = np.sum(nodeArea)*xspace**2/largestLung
        maxDia = np.max(maxDia)  
        #Get the mean entropy of all candinate nodes
        meanEn = np.min(En)
        #Features about attenuation of the node
        nodeAtt = nodeAtt[nodeAtt>=-10]
        if len(nodeAtt)>0:
            maxAtt = np.max(nodeAtt)
            stdAtt = np.std(nodeAtt)
            minAtt = np.min(nodeAtt)  
            meanAtt = np.mean(nodeAtt)
            #Compile all the features together
            features = np.asarray([numNode, totalVol, maxVol, minVol, maxAtt, stdAtt, meanAtt, \
                                   minAtt, minCirc, maxEqDia, stdEqDia, maxArea, avgArea, \
                                   totalArea, maxDia, dnodeX, dnodeY, maxfracConc, minAdg, meanEn]) 
        else:
            features = np.asarray([0,0,0,0,-500,-500,-500, \
                                   -500, 1, 0, 0, 0, 0, \
                                   0, 0, 256, 512, 0, 1, 10])
            contours = []
    else:
        features = np.asarray([0,0,0,0,-500,-500,-500, \
                               -500, 1, 0, 0, 0, 0, \
                               0, 0, 256, 512, 0, 1, 10])
        contours = []
    
#    plt.figure(figsize=(18,9),num=1)
#    plt.subplot(131)
#    plt.imshow(np.max(np.asarray(pat),axis=0))
#    plt.gca().add_patch(plt.Rectangle((leftLimit, toplimit), rightLimit-leftLimit, bottomlimit-toplimit, lw=2))
#    
#    plt.subplot(132)
#    plt.imshow(superMask)
#
#    mask = np.zeros(superMask.shape, np.uint8)        
#    mask = cv2.fillPoly(mask, contours, 255)
#    superMask[mask==0] = 0
#    
#    plt.subplot(133)
#    plt.imshow(superMask)
#    plt.show()
            
    return features

#process each patients DICOM images
def process_patient(patients, dicom_root, imgFolder='train'):

    feature_df = np.empty((0,29))
    
    for patient_no in patients:
        #Features that will be extracted from the CT scan for each patient
        largestLung = 0
        bloodArea = 0
        fatArea = 0
        waterArea = 0
        avgLung = 0
        stdLung = 0
        
        totalLung = 0 #use to normalize the bloodArea at each slice
        
        LungPix = np.empty((0,), dtype=np.uint8)  #Temp parameter for calculating avgLung and stdLung
        
        candNodes = [] #temp parameter to store the image array with node candidate masks
        lungImg = [] #arrays to store all segmented lungs for each patient
        
        pat = load_patient(patient_no, dicom_root)
        print(patient_no)
        
        xSpace = pat[0].PixelSpacing[0]*1.
        ySpace = pat[0].PixelSpacing[1]*1.
        try:
            zSpace = abs(pat[int(len(pat)/2)].SliceLocation*1. - pat[int(len(pat)/2)-1].SliceLocation*1.)
        except:
            zSpace = 2.
        
        ######Get lung segments###################
        for i in range(len(pat)):
            img = pat[i].image.copy()
        
            # threshold HU > -300
            img[img>-300] = 255
            img[img<-300] = 0
            img = np.uint8(img)
        
            # find surrounding torso from the threshold and make a mask
            im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(img.shape, np.uint8)
            mask = cv2.fillPoly(mask, [largest_contour], 255)
        
            # apply mask to threshold image to remove outside. this is our new mask
            img = ~img
            img[(mask == 0)] = 0 # <-- Larger than threshold value
            
            #remove the air blob in the middle
            im2, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)<1:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            for k in range(len(contours)-1,-1,-1):
                if len(contours[k])<4: #For those contours with only 4 or fewer points
                    contours.pop(k)
                    continue
            
                area = cv2.contourArea(contours[k]) #for very small areas such as air blob
                if (area < img.shape[0]*img.shape[1]/50):
                    contours.pop(k)
                    continue
                
                arcLength = cv2.arcLength(contours[k],True)
                roundness = (4*np.pi*area)/arcLength**2
                
                if (area/largest_area<0.25) and (roundness>0.5): #for round and small areas such as air blob
                    contours.pop(k) 
                    continue
                    
                M = cv2.moments(contours[k])
                Cx = M['m10']/M['m00']
                Cy = M['m01']/M['m00']
                if Cy<110 or Cy>400: #for the remaining segments not belonging to the lung
                    contours.pop(k) 
                    continue                    
                    
            if len(contours)==0: #when there is no lung segments found in current slice
                continue
                    
            #Seperate the lungs into left and right
            if len(contours)==1:
                mask = np.zeros(img.shape, np.uint8)
                mask = cv2.fillPoly(mask, contours, 255)
                contours = separate_lungs(contours, mask)  
            
            contours = sorted(contours, key=cv2.contourArea) #Keep the two largest lung segment; can be used to remove noise 
            numCnt = len(contours)
            while numCnt>2:            
                contours.pop(0)
                numCnt = len(contours)
                
            #further processing of the left and right lungs separately, in order to NOT include tissues in-between the two lung segments
            im2 = np.zeros(img.shape,np.uint8)
            for cnt in contours:
                mask = np.zeros(img.shape, np.uint8)
                mask = cv2.fillPoly(mask, [cnt], 255)
                #Closure operation with a disk of radius 21. This operation is 
                #to keep nodules attached to the lung wall.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                im2 += mask
            im2 = im2.clip(0,255)
            img = np.uint8(im2.copy())
            
            #Find the contours again
            im2, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
            
            #Need further to check if there is unwanted pieces
            for k in range(len(contours)-1,-1,-1):
                area = cv2.contourArea(contours[k])
                _,_,w,h=cv2.boundingRect(contours[k])
                if (area < img.shape[0]*img.shape[1]/50) or (w<5) or (h<5):
                    contours.pop(k)
                    
            #Seperate the lungs into left and right, in case the two lungs connected again after closing of each
            if len(contours)==1:
                mask = np.zeros(img.shape, np.uint8)
                mask = cv2.fillPoly(mask, contours, 255)
                contours = separate_lungs(contours, mask)
            
            #For the remaining lung segments, fill the indentation in case there are nodes at the lung border
            for k in range(len(contours)-1,-1,-1):
                contours[k] = fill_indentation(contours[k], 1.25)
            
            mask = np.zeros(img.shape, np.uint8)
            img = cv2.fillPoly(mask, contours, 255)
            
            # apply mask to image
            img2 = pat[i].image.copy()
            img2[(img == 0)] = -2000 # <-- Larger than threshold value  
    
            #Calculate the blood area, fat area, and water area
            bloodArea += xSpace*ySpace*np.sum((img2>=30) & (img2<=45))  
            fatArea += xSpace*ySpace*np.sum((img2>=-100) & (img2<=-50))  
            waterArea += xSpace*ySpace*np.sum((img2>=-5) & (img2<=5))  
            totalLung += xSpace*ySpace*np.sum(img2 != -2000)
                
            #Retrieve all the pixels in the lung area for later calculation of average and standard deviation of lungs
            LungPix = np.hstack((LungPix, img2[img2 != -2000].flatten()))
            
            #Put all lung segments into a list for later use
            lungImg.append(img2)
        
        #Get some key coordinates of the lung segments
        lungImg = np.asarray(lungImg)
        allLung = np.nanmax(lungImg, axis=0)   
        
        #get the bounding box for the largest lung
        tempLung = allLung.copy()
        tempLung[tempLung!=-2000] = 255
        tempLung[tempLung==-2000] = 0
        tempLung = np.uint8(tempLung)
        
        #Get the largest lung area for the patients
        largestLung = xSpace*ySpace*np.sum(allLung != -2000) 
        _,contours,_ = cv2.findContours(tempLung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            largest_contour = sorted(contours, key=cv2.contourArea)[-2:]
            largest_contour = np.vstack((largest_contour[0], largest_contour[1]))
        else:
            largest_contour = max(contours, key=cv2.contourArea)
        #center, Leftmost, rightmost, topmost, and bottommost point of the largest lung
        lungLeftx = largest_contour[largest_contour[:,:,0].argmin()][0][0]
        lungRightx = largest_contour[largest_contour[:,:,0].argmax()][0][0]
        lungTopy = largest_contour[largest_contour[:,:,1].argmin()][0][1]
        lungBottomy = largest_contour[largest_contour[:,:,1].argmax()][0][1]
        M = cv2.moments(largest_contour)
        lungCx = M['m10']/M['m00']
        lungCy = M['m01']/M['m00']
        lungBound = (lungCx, lungCy, lungLeftx, lungRightx, lungTopy, lungBottomy)    
        
        newLungImg = []   
        #####The following is used to find the nodes##################    
        for i in range(len(lungImg)):
            img = lungImg[i].copy()
            
            #get rid of the regions that are not nodes before feeding to unet
            img = remove_infarction(img)
        
            img2 = img.copy()
            ###Normalize the image##########
            mean = np.mean(img[img!=-2000])
            std = np.std(img[img!=-2000])
            img[img==-2000]=mean - 1.2*std
            img = img - mean
            img = img/(std+1e-8)
            new_mean = np.mean(img)
            img = img - new_mean
            minP = np.min(img)
            maxP = np.max(img)
            img = img/(maxP-minP+1e-8)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
            
            #get the node mask using the unet model
            nodeMask = unetmodel.predict(img, batch_size=1)[0,0,:,:]
            nodeMask[nodeMask>0.5] = 255
            nodeMask[nodeMask<=0.5] = 0
            nodeMask = np.uint8(nodeMask) 
            
            isNode = np.sum(nodeMask)
            if isNode==0:
                contours =[]
                lungSeg = img2
            else:
                _, contours, _ = cv2.findContours(nodeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours, lungSeg = remove_benign_nodes(contours, img2, lungBound, xSpace, ySpace)
                
            img3 = np.zeros(img2.shape, np.uint8) 
            img3 = cv2.fillPoly(img3, contours, 255)    
            candNodes.append(img3) 
            newLungImg.append(lungSeg)        
        
        #show the largest lung segments
        lungImg = np.asarray(newLungImg)
        allLung = np.nanmax(lungImg, axis=0)
        tempLung = allLung.copy()
        allLung += 2000         
        
        lungImg[lungImg==-2000] = np.nan
        stdLung = np.nanstd(lungImg, axis=0)
        stdLung[np.isnan(stdLung)] = 0   
        meanLung = np.nanmean(lungImg, axis=0)
        meanLung[np.isnan(meanLung)] = -1000
        meanLung = meanLung + 1000
        
        lung = np.zeros((allLung.shape[0], allLung.shape[1], 3))
        lung[:,:,0] = allLung
        lung[:,:,1] = meanLung    
        lung[:,:,2] = stdLung
        
        #save the alllung for later training in CNN
        if not os.path.isdir(imgFolder):
            os.mkdir(imgFolder)
        imsave(os.path.join(imgFolder, patient_no+'.jpg'), lung) 
        
        #get the bounding box for the largest lung
        tempLung[tempLung>-300] = 255
        tempLung[tempLung<-300] = 0        
        tempLung = np.uint8(tempLung)   
        
        _,contours,_ = cv2.findContours(tempLung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)        
        connContours = contours[-2:]  
        
        avgLung = np.nanmean(LungPix)
        maxLung = np.nanmax(LungPix)
        minLung = np.nanmin(LungPix)
        stdLung = np.nanstd(LungPix)
        tissueArea = (cv2.contourArea(connContours[0]) + cv2.contourArea(connContours[1]))*xSpace*ySpace/largestLung     
        bloodArea = bloodArea/largestLung
        fatArea = fatArea/largestLung
        waterArea = waterArea/largestLung
        
        #Now extract information about candidate nodes for each patient
        isNode = np.sum(candNodes)
        if isNode != 0:
            features = extract_node_features(candNodes, lungImg, largestLung, connContours, lungBound, xSpace, ySpace, zSpace)
        else:
            features = np.asarray([0,0,0,0,-500,-500,-500, \
                                   -500, 1, 0, 0, 0, 0, \
                                   0, 0, 256, 512, 0, 1, 10])
        features = np.hstack((features, [avgLung, maxLung, minLung, stdLung, bloodArea, fatArea, waterArea, tissueArea, largestLung])) 
        
        feature_df = np.vstack((feature_df, features))
        
    return feature_df   
