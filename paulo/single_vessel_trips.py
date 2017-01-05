"""
SINGLE VESSEL TRIP PACKAGE :
Every helper we could develop to plot a vessel's trip
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # mandatory to be able to export plots from remote server
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.basemap import Basemap
from sklearn import svm


def plot_vessel_trip(mmsi, after=None, before=None):

    # load positions table
    df = sqlContext.read.format(
        "org.apache.spark.sql.cassandra"
    ).options(
        table="basic_position",
        keyspace="ais_datas"
    ).load().createOrReplaceTempView("pos")

    # prepare the sql request
    request = "SELECT DISTINCT latitude, longitude "
    request += "FROM pos "
    request += "WHERE mmsi=" + str(mmsi)

    if after:
        pass

    if before:
        pass

    # launch
    query = spark.sql(request).collect()

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
    for row in query:
        # draw the corresponding point
        lon = float(row['longitude'])
        lat = float(row['latitude'])
        x, y = themap(lon, lat)
        themap.plot(x, y, 'bo', markersize=4)

    # finally save
    plt.savefig('./plots/' + str(mmsi) + '_trip.png')
