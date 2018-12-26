# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import preprocessing
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.decomposition import KernelPCA
import sklearn




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
y=data['label']

scikit_kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.15)
# 其中gamma的选值很重要，默认为1/n_features
X_skernpca = scikit_kpca.fit_transform(X)
print(X_skernpca)




X_train, X_test, y_train, y_test = train_test_split(X_skernpca, y, test_size = 0.50, random_state = 1)








tuned_parameters = [{'max_depth': [2,3,4,5],'silent': ['Ture','False'], 'seed':[0,1],'min_child_weight': [3,4,6,7],'objective':["binary:logistic"],
                     'colsample_bytree': [0.7,0.5,0.4,0.3],'gamma':[0.1,0.3,0.5,0.7],'learning_rate':[0.01,0.1,0.3],'n_estimators':[50,40,100,200],'booster':['gbtree','gblinear','dart']}]

#
# X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
#     X_train_minmax, Y_train, test_size=0.3, random_state=0)
scores = ['precision_macro', 'recall_macro','roc_auc','neg_log_loss']
# print(sorted(sklearn.metrics.SCORERS.keys()))
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=4,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    fh = open('best_params_xgb.txt', 'a')
    fh.write('best_params:'+json.dumps(clf.best_params_)+'\n')
    fh.close()
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        fh = open('best_params_xgb.txt', 'a')
        fh.write(str(mean)+str(std)+str(params) + '\n')
        fh.close()
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

