
# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import preprocessing
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,recall_score,roc_auc_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import random


data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']

X_train_origin=X.__deepcopy__()
X_train=X.__deepcopy__()
X_train2=X.__deepcopy__()

con=1.02
for i in range(X_train.shape[0]):
    [c,d,e,f,g,h,j,k]=[random.randint(0, 13) for _ in range(8)]
    X_train.iloc[i,:] = X_train.iloc[i,:].replace(X_train.iloc[i,c],X_train.iloc[i,c]*con)
    X_train.iloc[i,:] = X_train.iloc[i,:].replace(X_train.iloc[i,d],X_train.iloc[i,d]*con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, e], X_train.iloc[i, e] * con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, f], X_train.iloc[i, f] * con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, g], X_train.iloc[i, g] * con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, h], X_train.iloc[i, h] * con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, j], X_train.iloc[i, j] * con)
    X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, k], X_train.iloc[i, k] * con)

con2=1.02
for i in range(X_train2.shape[0]):
    [c,d,e,f,g,h,j,k]=[random.randint(0, 13) for _ in range(8)]
    X_train2.iloc[i,:] = X_train2.iloc[i,:].replace(X_train2.iloc[i,c],X_train2.iloc[i,c]*con2)
    X_train2.iloc[i,:] = X_train2.iloc[i,:].replace(X_train2.iloc[i,d],X_train2.iloc[i,d]*con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, e], X_train2.iloc[i, e] * con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, f], X_train2.iloc[i, f] * con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, g], X_train2.iloc[i, g] * con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, h], X_train2.iloc[i, h] * con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, j], X_train2.iloc[i, j] * con2)
    X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, k], X_train2.iloc[i, k] * con2)


X_train1= pd.concat([X_train,X_train_origin,X_train2],ignore_index=True)

y_train1=pd.concat([y,y,y],ignore_index=True)



X1=np.array(X_train1)



for i in range(10,100,1):
    for j in [0.5]:
        sum=0
        for k in range(100):
           scikit_kpca = KernelPCA(n_components=i, kernel='cosine',gamma=j)
           # 其中gamma的选值很重要，默认为1/n_features
           X_skernpca = scikit_kpca.fit_transform(X1)
           X_train5, X_test5, y_train5, y_test5 = train_test_split(X_skernpca, y_train1, test_size=0.25, random_state=k)
           clf1 = XGBClassifier()
           clf1.fit(X_train5, y_train5)
           y_true, y_pred = y_test5, clf1.predict(X_test5)
           sum+=roc_auc_score(y_true, y_pred)
        if sum/10>0.7:
           print(sum/10)
           print(i,j)