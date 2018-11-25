#!/usr/bin/env python
# -*-coding:utf-8-*-
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

spdata=pd.read_csv('spdata_10.csv')
print(spdata.shape)
data=[]
for spdata_line in spdata:
    data.append(spdata_line)
spdata=np.array(spdata)
spdata=spdata[:,1:7]
spdata=spdata.astype(np.float32)
scaler=MinMaxScaler()
spdata=scaler.fit_transform(spdata)
y_data=spdata[:,3:4]
train_x=np.zeros((2960,20,6))
train_y=np.zeros((2960,1))
test_x=np.zeros((300,20,6))
test_y=np.zeros((300,1))
for i in range(2960):
    train_x[i]=spdata[i:i+20,:]
    train_y[i]=y_data[i+20,:]
for i in range(3000,3300):
    test_x[i-3000]=spdata[i:i+20,:]
    test_y[i-3000]=y_data[i+20,:]


model=models.Sequential()
model.add(layers.LSTM(10,activation='tanh',recurrent_activation='hard_sigmoid',
                      recurrent_initializer='Orthogonal',
                      kernel_initializer='glorot_uniform',
                      input_shape=(None,6)))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.Adam(),loss='mae')
model.summary()
history=model.fit(train_x,train_y,batch_size=64,epochs=50,validation_data=(test_x,test_y))
model.save('testmodel1')

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.figure()
plt.show()