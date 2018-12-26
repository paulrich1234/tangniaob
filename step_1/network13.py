import pandas as pd
import numpy as np
# from keras import models
# from keras import layers
# from keras import optimizers
from sklearn.model_selection import train_test_split
# from keras import losses
# from keras import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)
print(len(X))
clf =MLPClassifier(hidden_layer_sizes=(50, ), activation='tanh', solver='adam', alpha=0.0001, batch_size=10,
learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
verbose=False, warm_start=False, momentum=0.5, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
clf.fit(np.array(X_train),y_train)
y_pred = clf.predict(np.array(X_test))
print(accuracy_score(y_pred,np.array(y_test)))


print(classification_report(np.array(y_test), y_pred))
print()
