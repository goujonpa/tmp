import numpy as np
from sklearn import svm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = sqlContext.read.format("org.apache.spark.sql.cassandra").options(table="basic_position", keyspace="ais_datas").load()
df.createOrReplaceTempView("vessels")
t = spark.sql("SELECT mmsi, timestamps, latitude, longitude FROM vessels WHERE timestamps >= \"2016-12-10 22:53:00\" AND timestamps < \"2016-12-10 22:54:00\" ")
t.show(10)
rddLat = t.rdd.map(lambda l:  l.latitude).collect()
rddLong = t.rdd.map(lambda l:  l.longitude).collect()
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
coord = [rddLat, rddLong]
clf.fit(coord)
plt.contour(X, Y, 
                        levels=[-9999], colors="k",
                        linestyles="solid")
plt.xticks([])
plt.yticks([])
plt.savefig('heatmap')
