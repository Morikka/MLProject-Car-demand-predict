# Based on LSTM Model

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Embedding
from keras.preprocessing import sequence
# import keras.backend as K
# from sklearn.preprocessing import MinMaxScaler

source = pd.read_csv('data/fake.csv')
  # dtype={'pickup_place':'category','dropoff_place':'category'})
# print(type(source))
source['pickup_date'] =  pd.to_datetime(source['pickup_date'])
# source['dropoff_date'] =  pd.to_datetime(source['dropoff_date'])

time_start = pd.to_datetime('2018-12-02 00')
trainData = source.where(source['pickup_date']<time_start)
trainData = trainData.dropna(how = 'any', axis = 'rows')
testData = source.where(source['pickup_date']>=time_start)
testData = testData.dropna(how = 'any', axis = 'rows')
# print(trainData)
trainData = trainData.to_numpy()
testData = testData.to_numpy()
x_train = trainData[:,1:4]
y_train = trainData[:,4]
x_test = testData[:,1:4]
y_test = testData[:,4]

max_features = 500

model = Sequential()
model.add(Embedding(max_features, output_dim=3))
model.add(LSTM(units=24))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

