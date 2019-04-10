import pandas as pd
import shapefile
import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# source = pd.read_csv('Data/yellow_tripdata_2018-12.csv', \
source = pd.read_csv('Data/temp.csv', \
	dtype={'VendorID':'category','RatecodeID':'category','store_and_fwd_flag':'category','PULocationID':'category','DOLocationID':'category','payment_type':'category'}, \
	parse_dates=["tpep_pickup_datetime","tpep_dropoff_datetime"])

print(source.info())
print("------------------------")
# list first few rows (datapoints)
print(source.head())
print("------------------------")
# check statistics of the features
print(source.describe())
print("------------------------")
print('Old size: %d' % len(source))
#remove negative data
for cols in source.columns.tolist()[10:]:
	print(cols)
	source = source[source[cols]>=0]
print('New size: %d' % len(source))
print("------------------------")
# check statistics of the features
print(source.describe())
print("------------------------")
print(source.isnull().sum())
print("------------------------")
print('Old size: %d' % len(source))
source = source.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(source))
print("------------------------")
print(source.head())
time1 = pd.Timestamp('2018-12-19 23:59:59')
print(type(time1))
trainData = source.where(source["tpep_pickup_datetime"]<time1)
testData = source.where(source["tpep_pickup_datetime"]>=time1)
print("------------------------")
print(trainData.describe())
print("------------------------")
print(testData.describe())
sf = shapefile.Reader("Data/taxi_zones/taxi_zones.shp")
reg_list={'Staten Island':1, 'Queens':2, 'Bronx':3, 'Manhattan':4, 'EWR':5, 'Brooklyn':6}
matplotlib.use('Agg') # no UI backend
plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y,'k')
plt.savefig("matplotlib.png")
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
# ax = plt.subplot(1, 2, 1)
# ax.set_title("Boroughs in NYC")