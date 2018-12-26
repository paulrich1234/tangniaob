from sklearn.manifold import Isomap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


embedding = Isomap(n_neighbors=7,n_components=2)



data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
X=np.array(X)

y=data['label']


X_skernpca = embedding.fit_transform(X)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))

ax[0].scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[1].scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_skernpca, y, test_size = 0.50, random_state = 1)
clf1=XGBClassifier()

clf1.fit(X_train,y_train)

y_true, y_pred = y_test, clf1.predict(X_test)

print(classification_report(y_true, y_pred))
print()
