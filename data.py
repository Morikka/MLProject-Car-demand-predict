import pandas as pd
import shapefile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
# print(sf.shapeType) -> POLYGON = 5
reg_list={'Staten Island':0, 'Queens':1, 'Bronx':2, 'Manhattan':3, 'EWR':4, 'Brooklyn':5}
colors = [(174,199,232),(255,187,120),(152,223,138),(255,152,150),(197,176,213),(196,156,148)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(colors)):    
  r, g, b = colors[i]    
  colors[i] = (r / 255., g / 255., b / 255.)
# print(reg_list['Queens'])
# matplotlib.use('Agg') # no UI backend
plt.figure()
for sr in sf.shapeRecords():
  shape = sr.shape
  rec = sr.record
  axes = plt.gca()
  color=colors[reg_list[rec[5]]]
  nparts = len(shape.parts) # total parts
  for ip in range(nparts):
    i0 = shape.parts[ip]
    if ip < nparts-1:
      i1 = shape.parts[ip+1]-1
    else:
      i1 = len(shape.points)
    # polygon = Polygon(shape.points[i0:i1+1])
    # patch = PolygonPatch(polygon, facecolor=col, alpha=1.0, zorder=2)
    # ax.add_patch(patch)
    x = [i[0] for i in shape.points[i0:i1+1]]
    y = [i[1] for i in shape.points[i0:i1+1]]
    axes.add_patch(Polygon(list(zip(x,y)),closed=True, facecolor=color))
    plt.plot(x,y,'b',linewidth=0.2)
plt.show()
# plt.savefig("matplotlib.png")
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
# ax = plt.subplot(1, 2, 1)
# ax.set_title("Boroughs in NYC")
