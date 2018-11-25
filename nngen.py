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

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=32,step=1):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arrange(i,min(i+batch_size,max_index))
            i+=len(rows)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][3]
        yield samples,targets

spdata=pd.read_csv('spdata_10.csv')
print(spdata.shape)
data=[]

for spdata_line in spdata:
    data.append(spdata_line)
spdata=np.array(spdata)
spdata1=spdata[0:3301,1:7]
spdata2=spdata[3302:-1,1:7]

spdata1=spdata1.astype(np.float32)
spdata2=spdata2.astype(np.float32)
y_data1=spdata1[:,3:4]
y_data2_1=spdata2[:,3:4]
miny=min(y_data2_1)
maxy=max(y_data2_1)

scaler=MinMaxScaler()
spdata1=scaler.fit_transform(spdata1)
spdata2=scaler.fit_transform(spdata2)
y_data2=spdata2[:,3:4]

test_days=400
test_x=np.zeros((test_days,20,6))
test_y=np.zeros((test_days,1))
for i in range(0,test_days):
    test_x[i]=spdata2[i:i+20,:]
    test_y[i]=y_data2[i+20,:]


lookback=20
delay=1
step=1
batch_size=32
train_gen=generator(spdata1,lookback=lookback,delay=delay,min_index=0,max_index=3000,shuffle=True,step=step,
                    batch_size=batch_size)
val_gen=generator(spdata1,lookback=lookback,delay=delay,min_index=3001,max_index=3300,shuffle=True,step=step,
                  batch_size=batch_size)
test_gen=generator(spdata,lookback=lookback,delay=delay,min_index=3301,max_index=None,shuffle=True,step=step,
                   batch_size=batch_size)

val_steps=(3300-3001-lookback)//batch_size
test_steps=(len(spdata)-3001)//batch_size

model=models.Sequential()
model.add(layers.LSTM(10,activation='tanh',recurrent_activation='hard_sigmoid',
                      recurrent_initializer='Orthogonal',
                      kernel_initializer='glorot_uniform',
                      input_shape=(None,6)))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.Adam(),loss='mae')
model.summary()
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)
model.save('testmodel2.h5')


loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

y=model.predict(test_x)


y=y*(maxy-miny)+miny
test_y=test_y*(maxy-miny)+miny
print(y)
print(test_y)
print(abs(test_y-y))
print(y.shape)
print(test_y.shape)
mae=0

for i in range(len(y)):
    mae=mae+abs(y[i,0]-test_y[i,0])

ploty=np.reshape(y,len(y))
plotty=np.reshape(test_y,len(y))
print(mae/len(y))
plt.plot(ploty,'b',label='perdict')
plt.plot(plotty,'r',label='real')
plt.title('value')
plt.show()