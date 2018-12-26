import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import random






data=pd.read_table('../combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train_origin=X_train.__deepcopy__()
X_train2=X_train.__deepcopy__()

con=1.05
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

con2=1.05
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


X_train= pd.concat([X_train,X_train_origin,X_train2],ignore_index=True)
y_train=pd.concat([y_train,y_train,y_train],ignore_index=True)





# print(X_train)

model = models.Sequential()
model.add(layers.Dense(7, activation='hard_sigmoid', input_shape=(14,)))
model.add(layers.Dense(7, activation='hard_sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))





model.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(np.array(X_train),
                    y_train,
                    epochs=2550,
                    batch_size=20,
                    validation_data=(np.array(X_test), y_test))

history_dict = history.history
history_dict.keys()


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()



plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()