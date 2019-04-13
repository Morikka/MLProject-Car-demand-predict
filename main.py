import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

# source = pd.read_csv('Data/temp.csv',header=1)
source = pd.read_csv('Data/temp.csv')
# print(source.info())
model = Sequential()
model.add(Dense(32, input_dim=17))
model.add(Activation('relu'))

train = ['VendorID','tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance','RatecodeID','store_and_fwd_flag','PULocationID','DOLocationID','payment_type','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount']

x_train = source[train]
y_train = source[]
# model.fit(x_train, y_train, epochs=5, batch_size=32)
