import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

sample = 28
trainsample = 26
time = 24
testtime = 20
size1 = 40
size2 = 40
batchsize = 10

data = np.zeros((sample,time,size1,size2,1))

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='mean_squared_error', optimizer='adadelta')

def getData():
  source = pd.read_csv('data/temp.csv', parse_dates=["pickup_datetime"])
  print("---finish import---")
  source = source[['pickup_datetime','pickup_longitude','pickup_latitude']]
  lon_max = source['pickup_longitude'].max()
  lon_min = source['pickup_longitude'].min()
  lat_max = source['pickup_latitude'].max()
  lat_min = source['pickup_latitude'].min()
  source = source.groupby(source['pickup_datetime'].dt.date)
  print("---finish data processing---")
  step1 = 0
  step2 = 0
  for time,tmp in source: #for every day (sample)
    tmp = tmp.assign(hour = lambda x: x['pickup_datetime'].dt.hour)
    tmp = tmp[['hour','pickup_longitude','pickup_latitude']]
    tmp = tmp.groupby(tmp['hour'])
    step2 = 0
    for hour, item in tmp: #for every hour
      item = item.to_numpy()
      for i in item: # for every item in hour
        lon = i[1]
        lat = i[2]
        lon = (size1-1) * (lon - lon_min) / (lon_max - lon_min)
        lat = (size2-1) * (lat - lat_min) / (lat_max - lat_min)
        lon = int(lon)
        lat = int(lat)
        data[step1][step2][lon][lat][0] += 1
      step2 += 1
    step1 += 1
  print("Step 1 is",step1)
  print("---finish data counting---")
  return data

def main():
  data = getData()

  #Train the network - use 26 days(or can be regarded as 27 days train 26 times)
  newdata = data[1:trainsample]
  olddata = data[:trainsample-1]
  seq.fit(olddata,newdata, batch_size=batchsize, epochs=3, validation_split=0.05)
  print("---finish training---")

  #Test the network - use the last day
  testdata = data[sample-1]
  # use the first 20 hours to get the last 4
  track = testdata[:testtime,::,::,::]
  print(">>>>>><><>",track.shape)

  for i in range(time-testtime):
    print("The time",i)
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
    print(">>>>>>>>>>",track.shape)
    print(new_pos.shape)
    print(new.shape)
    print(testdata[:(testtime+i)].shape)
    print(testdata[:(testtime+i)][np.newaxis, ::, ::, ::, ::].shape)
    score = seq.evaluate(track[np.newaxis, ::, ::, ::, ::],testdata[:(testtime+i+1)][np.newaxis, ::, ::, ::, ::], batch_size=batchsize, verbose=1)
    print(score)

  # and compare the preditions later


if __name__ == '__main__':
  main()
