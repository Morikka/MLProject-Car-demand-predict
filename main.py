import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

# source = pd.read_csv('Data/temp.csv',header=1)
source = pd.read_csv('Data/temp.csv')
# print(source.info())
model = Sequential()
model.add(Dense(32, input_dim=17))
model.add(Activation('relu'))

train = ['pickup_date','pickup_hour','dropoff_date','dropoff_hour','pickup_place','dropoff_place','number']
x_train = source[train]
y_train = source[]
# model.fit(x_train, y_train, epochs=5, batch_size=32)
