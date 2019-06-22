# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:20:04 2017

1) Train, validate, and test the model
2) Write the submission files

@author: linhaili
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss as logloss

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ETC 
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from scipy.stats import gmean

#combine several classifier into a single one
def combine_classifier(clfs):
    clf = clfs[0]
    for kclf in range(1,len(clfs)):
        clf.estimators_ += clfs[kclf].estimators_
        clf.n_estimators = len(clf.estimators_)
        
    return clf

#use gmean to determine the final prediction
def gmean_pred(clfs, X):
    preds = []
    for clf in clfs:
        preds.append(clf.predict_proba(X)[:,1])
    pred = gmean(np.array(preds), axis=0).clip(0.001,1)
    
    return pred

#blend model by voting
def blend_prediction(predDF, cancerCols, threshold=0.5):
    raw_pred = predDF[cancerCols].values
    numPos = np.sum(raw_pred>threshold, axis=1)
    numNeg = np.sum(raw_pred<=threshold, axis=1)
    maxProb = np.max(raw_pred, axis=1)
    minProb = np.min(raw_pred, axis=1)
    
    pred = np.zeros((len(predDF),))
    pred[np.where((numPos-numNeg)>0)] = maxProb[np.where((numPos-numNeg)>0)]
    pred[np.where((numPos-numNeg)<=0)] = minProb[np.where((numPos-numNeg)<=0)]
    pred = pred.clip(0.0001, 1)
    
    return pred

#blend model by logistic classification by treating the predictions as features
from sklearn.linear_model import LogisticRegression
def blend_classification(predDF, cancerCols, truePred):
    raw_pred = predDF[cancerCols].values
    blf = LogisticRegression(C=0.8, n_jobs=6, random_state=8888)
    blf.fit(raw_pred, truePred)
    
    return blf

################Name of the CSV files to use################
dfName_train = 'train_features_unet.csv'
dfName_test = 'test_features_unet.csv'
dfName_sub = 'stage2_features_unet.csv'

##For those without a node, need to update the features
##Feature columns
#columns = ['numNode', 'maxHU', 'stdHU', 'meanHU', 'minHU', 'minCirc', \
#       'maxEqDia', 'stdEqDia', 'maxArea', 'avgArea', 'totalArea', 'maxDia', 'dnodeX', 'dnodeY', 'FracConc', \
#       'Acutance', 'Entropy'] 
#featureNoNode = np.asarray([0,-300,-300,-300,-300,1,0,0,0,0,0,0,0.5,0.9,0,1,20]).reshape((1,-1))

#Training data
train_data = pd.read_csv(dfName_train)
train_patient = train_data['id']

col = [col for col in train_data.columns if col not in ['id', 'cancer']]
col1 = col[0:29]
col = col[0:29]
X = train_data[col].values
Y = train_data['cancer'].values
              
#Validation data
val_data = pd.read_csv(dfName_test)

Xval = val_data[col].values
Yval = val_data['cancer'].values
                
# Stage 2 Test data
test_data = pd.read_csv(dfName_sub)

Xtest = test_data[col].values

###############################Random Forest##############################
print('#######################Random Forest########################')
kf = KFold(Y, n_folds=5, shuffle=True, random_state=884)
y_pred = Y * 0
y_pred_prob = Y * 0

clfs1 = []
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]   
    clf = RF(n_estimators=100, n_jobs=7)
    clf.fit(X_train, y_train)
    clfs1.append(clf)
    y_pred_test = clf.predict(X_test)
    y_pred_prob_test = clf.predict_proba(X_test)[:,1]
    print("Iteration - logloss",logloss(Y[test], y_pred_prob_test))

clf1 = combine_classifier(clfs1)    
#Training performance evaluation    
y_pred = clf1.predict(X)
y_pred_prob = clf1.predict_proba(X)[:,1]
print('Training Results:')
print(confusion_matrix(Y, y_pred))
print("logloss",logloss(Y, y_pred_prob))

train_data['RF_pred'] = clf1.predict_proba(X)[:,1]

#Testing performance evaluation    
y_pred1 = clf1.predict(Xval)
y_pred_prob1 = clf1.predict_proba(Xval)[:,1]
print('Testing Results:')
print(confusion_matrix(Yval, y_pred1))
print("logloss",logloss(Yval, y_pred_prob1))

#Estimation on testing data
y_pred_prob = clf1.predict_proba(Xtest)[:,1]
test_data['RF_pred'] = y_pred_prob   

##############################XGBoost##################################
print ("\n ######################XGBoost############################")
y_pred = Y * 0
y_pred_prob = Y * 0
clfs2 = []
result = []
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
    clf = xgb.XGBClassifier(max_depth=5, objective="binary:logistic", scale_pos_weight=3, \
                            n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8, \
                            subsample=0.85, colsample_bytree=0.9, seed=96)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, eval_metric='logloss', \
                                        early_stopping_rounds=100)
    clfs2.append(clf)
    result.append(clf.best_score)
    y_pred_test = clf.predict(X_test)
    y_pred_prob_test = clf.predict_proba(X_test)[:,1]
    print("Iteration - logloss",logloss(Y[test], y_pred_prob_test))

#Training performance evaluation      
y_pred = gmean_pred(clfs2, X)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<=0.5] = 0
y_pred_prob = gmean_pred(clfs2, X)
print('Training Results:')
print(confusion_matrix(Y, y_pred))
print("logloss",logloss(Y, y_pred_prob))

train_data['XGB_pred'] = gmean_pred(clfs2, X)

#Testing performance evaluation    
y_pred2 = gmean_pred(clfs2, Xval)
y_pred2[y_pred2>0.5] = 1
y_pred2[y_pred2<=0.5] = 0
y_pred_prob2 = gmean_pred(clfs2, Xval)
print('Testing Results:')
print(confusion_matrix(Yval, y_pred2))
print("logloss",logloss(Yval, y_pred_prob2))

#Estimation on testing data
y_pred_prob = gmean_pred(clfs2, Xtest)
test_data['XGB_pred'] = y_pred_prob 
    
#######################ExtraTrees################################
print('\n ################Extra Trees########################')
y_pred = Y * 0
y_pred_prob = Y * 0

clfs3 = []
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]   
    clf = ETC(n_estimators=100, n_jobs=7)
    clf.fit(X_train, y_train)
    clfs3.append(clf)
    y_pred_test = clf.predict(X_test)
    y_pred_prob_test = clf.predict_proba(X_test)[:,1]
    print("Iteration - logloss",logloss(Y[test], y_pred_prob_test))

clf3 = combine_classifier(clfs3)    
#Training performance evaluation    
y_pred = clf3.predict(X)
y_pred_prob = clf3.predict_proba(X)[:,1]
print('Training Results:')
print(confusion_matrix(Y, y_pred))
print("logloss",logloss(Y, y_pred_prob))

train_data['ETC_pred'] = clf3.predict_proba(X)[:,1]

#Testing performance evaluation    
y_pred3 = clf3.predict(Xval)
y_pred_prob3 = clf3.predict_proba(Xval)[:,1]
print('Testing Results:')
print(confusion_matrix(Yval, y_pred3))
print("logloss",logloss(Yval, y_pred_prob3))

#Estimation on testing data
y_pred_prob = clf3.predict_proba(Xtest)[:,1]
test_data['ETC_pred'] = y_pred_prob        
         
#######################ExtraTrees Warm Start################################
print('\n ################Extra Trees Warm Start########################')
y_pred = Y * 0
y_pred_prob = Y * 0

clf4 = ETC(n_estimators=20, warm_start=True, n_jobs=7)
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]       
    clf4.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_prob_test = clf.predict_proba(X_test)[:,1]
    clf4.n_estimators += 20
    print("Iteration - logloss",logloss(Y[test], y_pred_prob_test))
   
#Training performance evaluation    
y_pred = clf4.predict(X)
y_pred_prob = clf4.predict_proba(X)[:,1]
print('Training Results:')
print(confusion_matrix(Y, y_pred))
print("logloss",logloss(Y, y_pred_prob))

train_data['ETC_pred2'] = clf4.predict_proba(X)[:,1]

#Testing performance evaluation    
y_pred4 = clf4.predict(Xval)
y_pred_prob4 = clf4.predict_proba(Xval)[:,1]
print('Testing Results:')
print(confusion_matrix(Yval, y_pred4))
print("logloss",logloss(Yval, y_pred_prob4))

#Estimation on testing data
y_pred_prob = clf4.predict_proba(Xtest)[:,1]
test_data['ETC_pred2'] = y_pred_prob     
         
###############################Random Forest Warm Start##############################
print('#######################Random Forest Warm Start########################')
y_pred = Y * 0
y_pred_prob = Y * 0

clf5 = RF(n_estimators=20, warm_start=True, n_jobs=7)
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]       
    clf5.fit(X_train, y_train)
    y_pred_test = clf5.predict(X_test)
    y_pred_prob_test = clf5.predict_proba(X_test)[:,1]
    clf5.n_estimators += 20
    print("Iteration - logloss",logloss(Y[test], y_pred_prob_test))
    
#Training performance evaluation    
y_pred = clf5.predict(X)
y_pred_prob = clf5.predict_proba(X)[:,1]
print('Training Results:')
print(confusion_matrix(Y, y_pred))
print("logloss",logloss(Y, y_pred_prob))

train_data['RF_pred2'] = clf5.predict_proba(X)[:,1]

#Testing performance evaluation    
y_pred5 = clf5.predict(Xval)
y_pred_prob5 = clf5.predict_proba(Xval)[:,1]
print('Testing Results:')
print(confusion_matrix(Yval, y_pred5))
print("logloss",logloss(Yval, y_pred_prob5))

#Estimation on testing data
y_pred_prob = clf5.predict_proba(Xtest)[:,1]
test_data['RF_pred2'] = y_pred_prob   

#blend all predictions 
predDF = {'RF_pred': y_pred_prob1, 'XGB_pred': y_pred_prob2, 'ETC_pred': y_pred_prob3, 'ETC_pred2': y_pred_prob4, 'RF_pred2': y_pred_prob5}
predDF = pd.DataFrame.from_dict(predDF) 
cols = ['RF_pred', 'XGB_pred', 'ETC_pred', 'ETC_pred2', 'RF_pred2']
blPred1 = blend_prediction(predDF, cols, threshold=0.5).clip(0.0001, 1)       
blf = blend_classification(predDF, cols, Yval) 
blPred2 = blf.predict_proba(predDF[cols].values)[:,1].clip(0.0001, 1)    
loss1 = logloss(Yval, blPred1)
loss2 = logloss(Yval, blPred2)
print('loss1: {:.4f}, loss2: {:.4f}'.format(loss1, loss2))

#blend the predictions for stage 2 test data set and submit########################
blPred1 = blend_prediction(test_data, cols, threshold=0.5).clip(0.0001, 1)   
blPred2 = blf.predict_proba(test_data[cols].values)[:,1].clip(0.0001, 1) 
test_data['pred1'] = blPred1
test_data['pred2'] = blPred2

submission1 = pd.read_csv('stage2_sample_submission.csv')         
submission1['cancer'] = test_data['pred1'].values
submission2 = submission1       
submission2['cancer'] = test_data['pred2'].values
submission1.to_csv('stage2_submission_vote_20170410.csv', index=False, float_format='%.6f')
submission2.to_csv('stage2_submission_blend_20170410.csv', index=False, float_format='%.6f')
