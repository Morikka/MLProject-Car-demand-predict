import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

# source = pd.read_csv('Data/temp.csv',header=1)
source = pd.read_csv('Data/temp.csv')
# print(source.info())

# list first few rows (datapoints)
source.head()

model.add(Dense(32, input_dim=17))
model.add(Activation('relu'))

