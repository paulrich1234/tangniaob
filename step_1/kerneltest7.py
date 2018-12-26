
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
for j in range(16,70,1):
    sum = 0.0
    for k in range(1,10,1):
       print('iteration time :',k)
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=k)
       X_train_origin = X_train.__deepcopy__()
       con = 1.01
       for i in range(X_train.shape[0]):
          [c, d, e, f, g, h, j, k] = [random.randint(0, 13) for _ in range(8)]
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, c], X_train.iloc[i, c] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, d], X_train.iloc[i, d] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, e], X_train.iloc[i, e] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, f], X_train.iloc[i, f] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, g], X_train.iloc[i, g] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, h], X_train.iloc[i, h] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, j], X_train.iloc[i, j] * con)
          X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, k], X_train.iloc[i, k] * con)

       X_train1 = pd.concat([X_train, X_train_origin],ignore_index=True)
       print(X_train1)

       y_train1 = pd.concat([y_train, y_train], ignore_index=True)
       data_combined1 = pd.concat([X_train1, y_train1], axis=1)
       print(data_combined1)
       data_combined1.to_csv('combined1.01.csv')

       scikit_kpca = KernelPCA(n_components=j, kernel='cosine',gamma=0.5)


       X_train_final = scikit_kpca.fit_transform(np.array(X_train1))

       X_test_final = scikit_kpca.transform(np.array(X_test))

       clf1 = XGBClassifier()
       clf1.fit(X_train_final, y_train1)
       y_true, y_pred = y_test, clf1.predict(X_test_final)
       sum+=recall_score(y_true, y_pred)
    if sum/10.0>0.7:
        print('10 times average score:',sum/10)
        print('components :',j)