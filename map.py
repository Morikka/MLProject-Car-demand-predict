import pandas as pd
import shapefile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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
  print(rec)
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
