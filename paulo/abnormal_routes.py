"""
ABNORMAL ROUTES :
Every helper we could develop to plot the abnormal
"""

import numpy as np
import json
import datetime
import matplotlib
matplotlib.use('Agg')  # mandatory to be able to export plots from remote server
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.basemap import Basemap
from sklearn import svm


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# A N'UTILISER QU'UNE FOIS EN DEBUT D'UTILISATION

# def load_dest_data():
    """LOAD DATAS"""

# load tables and create a view
positions = sqlContext.read.format(
    "org.apache.spark.sql.cassandra"
).options(
    table="basic_position",
    keyspace="ais_datas"
).load().createOrReplaceTempView("pos")

destinations = sqlContext.read.format(
    "org.apache.spark.sql.cassandra"
).options(
    table="basic_destination",
    keyspace="ais_datas"
).load().createOrReplaceTempView("dest")

d = spark.sql("SELECT * FROM dest")
p = spark.sql("SELECT * FROM pos")

return (dest, pos)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def plot_all_dest(destination, pos, dest):
    """PLOTS EVERY POINT GOING TO A DEST"""
dest = d.filter(d.destination==destination).collect()
rows = list()
for row in dest:
    print(row)
    mmsi = row['mmsi']
    date_to = row['timestamps'].date()
    date_from = row['timestamps'].date() - datetime.timedelta(days=1)
    rows += p.filter(
        p.mmsi==mmsi
    ).filter(
        p.timestamps >= date_from
    ).filter(
        p.timestamps < date_to
    ).collect()

# plot
fig = plt.figure()

# create the map background
themap = Basemap(
    projection='gall',
    llcrnrlon=-4,              # lower-left corner longitude
    llcrnrlat=26,               # lower-left corner latitude
    urcrnrlon=37,               # upper-right corner longitude
    urcrnrlat=48,               # upper-right corner latitude
    resolution='l',
    area_thresh=100000.0,
)
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color='gainsboro')
themap.drawmapboundary(fill_color='steelblue')

# for raw in the returned dataset
for row in rows:
    # draw the corresponding point
    lon = float(row['longitude'])
    lat = float(row['latitude'])
    x, y = themap(lon, lat)
    themap.plot(x, y, 'bo', markersize=4)

# finally save
plt.savefig('./plots/' + str(destination) + '.png')

# DENSITIES
data = np.array([[row['longitude'], row['latitude']] for i in range(0, len(rows))])


OCS = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
OCS.fit(data)
xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50))
Z = OCS.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title(str(destination) + " : Routes reccurentes")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
plt.scatter(data[:, 0], data[:, 1], c='gold', s=40)
plt.axis('tight')
plt.xlim((-4, 37))
plt.ylim((26, 48))
plt.savefig('./plots/' + str(destination) + '_reccurent.png')
