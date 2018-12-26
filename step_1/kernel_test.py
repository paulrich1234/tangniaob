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




# # 训练数据标准化
# min_max_scaler = preprocessing.MinMaxScaler()
# data=pd
# X_train=train.drop(['label'],axis=1)
# Y_train=train['label']
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# # train_data =pd.concat([a,Y_train],axis=1,join_axes=[a.index])
# # print(train_data)

data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
X=np.array(X)

y=data['label']

for i in range(20,140,1):
    for j in [0.5]:
        sum=0
        for k in range(100):
           scikit_kpca = KernelPCA(n_components=i, kernel='cosine',gamma=j,degree=5)
# 其中gamma的选值很重要，默认为1/n_features
           X_skernpca = scikit_kpca.fit_transform(X)

           X_train, X_test, y_train, y_test = train_test_split(X_skernpca, y, test_size=0.20, random_state=k)
           clf1 = XGBClassifier()
           clf1.fit(X_train, y_train)
           y_true, y_pred = y_test, clf1.predict(X_test)
           sum+=recall_score(y_true, y_pred)
           # if accuracy_score(y_true, y_pred)>0.8:
           # print(i,j,k)
           # print(classification_report(y_true, y_pred))
           # print()
        if sum/100>0.7:
           print(sum/100)
           print(i,j)

# fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
#
# ax[0].scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
# ax[1].scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
#
# plt.tight_layout()
# plt.show()

