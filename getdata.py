#!/usr/bin/env python
# -*-coding:utf-8-*-
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
spdata=spdata[:,1:7]
spdata=spdata.astype(np.float32)
y_data=spdata[:,3:4]
print(min(y_data))
print(max(y_data))
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
lookback=20
delay=1
step=1
batch_size=32
train_gen=generator(spdata,lookback=lookback,delay=delay,min_index=0,max_index=3000,shuffle=True,step=step,
                    batch_size=batch_size)
val_gen=generator(spdata,lookback=lookback,delay=delay,min_index=3001,max_index=3300,shuffle=True,step=step,
                  batch_size=batch_size)
test_gen=generator(spdata,lookback=lookback,delay=delay,min_index=3301,max_index=None,shuffle=True,step=step,
                   batch_size=batch_size)
val_steps=(3300-3001-lookback)//batch_size
test_steps=(len(spdata)-300001)//batch_size



