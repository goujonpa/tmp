import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

df = sqlContext.read.format("org.apache.spark.sql.cassandra").options(table="basic_destination", keyspace="ais_datas").load()
df.createOrReplaceTempView("dest")
t = spark.sql("SELECT mmsi, date_format(timestamps, 'YYYY-MM-DD') as ts FROM dest WHERE destination='ROTTERDAM'")
t.createOrReplaceTempView("keydest")
df2 = sqlContext.read.format("org.apache.spark.sql.cassandra").options(table="basic_position", keyspace="ais_datas").load()
df2.createOrReplaceTempView("vessels")
res = spark.sql("SELECT latitude, longitude FROM vessels v LEFT JOIN keydest k ON v.mmsi=k.mmsi AND date_format(v.timestamps, 'YYYY-MM-DD') = k.ts")