import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

source = pd.read_csv('temp.csv')
  # dtype={'pickup_place':'category','dropoff_place':'category'})
# print(type(source))
source['pickup_date'] =  pd.to_datetime(source['pickup_date'])
source['dropoff_date'] =  pd.to_datetime(source['dropoff_date'])

time_start = pd.to_datetime('2019-12-21')
trainData = source.where(source['pickup_date']<time_start)
trainData = trainData.dropna(how = 'any', axis = 'rows')
testData = source.where(source['pickup_date']>=time_start)
testData = testData.dropna(how = 'any', axis = 'rows')
# print()
x_train = trainData.iloc[:,2:6]
y_train = trainData.iloc[:,6]
x_test = testData.iloc[:,2:6]
y_test = testData.iloc[:,6]
model = Sequential()

model.add(Dense(units=8, activation='relu', input_dim=4))
model.add(Dense(units=1, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
