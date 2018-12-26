# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,recall_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random


#读取数据
# data =pd.read_excel('origin_data.xlsx')
# X=data.drop(['糖尿病'],axis=1)

# y=data['糖尿病']
data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']
# print(X)
# print(y)
average_score =0

for s in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train_origin=X_train.__deepcopy__()
    X_train2=X_train.__deepcopy__()
    X_train3=X_train.__deepcopy__()
    con=1.05
    for i in range(X_train.shape[0]):
      [c,d,e,f,g,h,j,k]=[random.randint(0, 13) for _ in range(8)]
      X_train.iloc[i,:] = X_train.iloc[i,:].replace(X_train.iloc[i,c],X_train.iloc[i,c]*con)#按照行加数据
      X_train.iloc[i,:] = X_train.iloc[i,:].replace(X_train.iloc[i,d],X_train.iloc[i,d]*con)
      X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, e], X_train.iloc[i, e] * con)
      X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, f], X_train.iloc[i, f] * con)
      X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, g], X_train.iloc[i, g] * con)
      # X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, h], X_train.iloc[i, h] * con)
      # X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, j], X_train.iloc[i, j] * con)
      # X_train.iloc[i, :] = X_train.iloc[i, :].replace(X_train.iloc[i, k], X_train.iloc[i, k] * con)

    con2=1.05
    for i in range(X_train2.shape[0]):
      [c,d,e,f,g,h,j,k]=[random.randint(0, 13) for _ in range(8)]
      X_train2.iloc[i,:] = X_train2.iloc[i,:].replace(X_train2.iloc[i,c],X_train2.iloc[i,c]*con2)#按照行加数据
      X_train2.iloc[i,:] = X_train2.iloc[i,:].replace(X_train2.iloc[i,d],X_train2.iloc[i,d]*con2)
      X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, e], X_train2.iloc[i, e] * con2)
      X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, f], X_train2.iloc[i, f] * con2)
      X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, g], X_train2.iloc[i, g] * con2)
      # X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, h], X_train2.iloc[i, h] * con2)
      # X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, j], X_train2.iloc[i, j] * con2)
      # X_train2.iloc[i, :] = X_train2.iloc[i, :].replace(X_train2.iloc[i, k], X_train2.iloc[i, k] * con2)


    X_train= pd.concat([X_train,X_train_origin,X_train2,X_train3],ignore_index=True)
    y_train=pd.concat([y_train,y_train,y_train,y_train],ignore_index=True)
    clf1=XGBClassifier(colsample_bytree= 0.5, gamma= 0.2, learning_rate= 0.6, max_depth= 2, min_child_weight= 3,criterion= "entropy",min_samples_split=2,
                  n_estimators= 100, objective= "binary:logistic", seed= 1)

    clf1.fit(X_train,y_train)

    y_true, y_pred = y_test, clf1.predict(X_test)

    print(classification_report(y_true, y_pred))
    average_score+=accuracy_score(y_true,y_pred)
    print(accuracy_score(y_true,y_pred))



print(average_score/100)