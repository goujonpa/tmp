import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

df = sqlContext.read.format("org.apache.spark.sql.cassandra").options(table="basic_position", keyspace="ais_datas").load()
df.createOrReplaceTempView("vessels")
t = spark.sql("SELECT mmsi, timestamps, latitude, longitude FROM vessels WHERE timestamps >= \"2016-12-10 22:53:00\" AND timestamps < \"2016-12-10 22:54:00\" ")
t.createOrReplaceTempView("unique_time")
coo = spark.sql("SELECT latitude, longitude FROM unique_time")
dfLat = coo.rdd.map(lambda l:  l.latitude).collect()
dfLong = coo.rdd.map(lambda l:  l.longitude).collect()
data = np.array([[dfLong[i], dfLat[i]] for i in range(0,len(dfLat)) ]) 

########################################################################################################
#RBF kernel
rbf_clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
rbf_clf.fit(data)

xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50)) #Lat et long max de la grille
# plot the line, the points, and the nearest vectors to the plane
rbf_Z = rbf_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
rbf_Z = rbf_Z.reshape(xx.shape)

plt.title("Heat map")
plt.contourf(xx, yy, rbf_Z, levels=np.linspace(rbf_Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, rbf_Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, rbf_Z, levels=[0, rbf_Z.max()], colors='palevioletred')

pt = plt.scatter(data[:, 0], data[:, 1], c='gold', s=40) #Trace les points : indiquer long et lat de chaque point
#c:color et s:forme
plt.axis('tight')
#Changer les axes du graphique ici : 
plt.xlim((-4, 37)) 
plt.ylim((26, 48))
plt.savefig('git/datatreatment/HeatMaps/heatmapRBF.png')

########################################################################################################

#RBF kernel --> 0.5 nu
rbf_clf2 = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
rbf_clf2.fit(data)

xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50)) #Lat et long max de la grille
# plot the line, the points, and the nearest vectors to the plane
rbf_Z2 = rbf_clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
rbf_Z2 = rbf_Z2.reshape(xx.shape)

plt.title("Heat map")
plt.contourf(xx, yy, rbf_Z, levels=np.linspace(rbf_Z2.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, rbf_Z2, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, rbf_Z2, levels=[0, rbf_Z2.max()], colors='palevioletred')

pt = plt.scatter(data[:, 0], data[:, 1], c='gold', s=40) #Trace les points : indiquer long et lat de chaque point
#c:color et s:forme
plt.axis('tight')
#Changer les axes du graphique ici : 
plt.xlim((-4, 37)) 
plt.ylim((26, 48))
plt.savefig('git/datatreatment/HeatMaps/heatmapRBF2.png')

########################################################################################################

#RBF kernel --> auto gamma
rbf_clf3 = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma="auto")
rbf_clf3.fit(data)

xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50)) #Lat et long max de la grille
# plot the line, the points, and the nearest vectors to the plane
rbf_Z3 = rbf_clf3.decision_function(np.c_[xx.ravel(), yy.ravel()])
rbf_Z3 = rbf_Z3.reshape(xx.shape)

plt.title("Heat map")
plt.contourf(xx, yy, rbf_Z, levels=np.linspace(rbf_Z3.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, rbf_Z3, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, rbf_Z3, levels=[0, rbf_Z3.max()], colors='palevioletred')

pt = plt.scatter(data[:, 0], data[:, 1], c='gold', s=40) #Trace les points : indiquer long et lat de chaque point
#c:color et s:forme
plt.axis('tight')
#Changer les axes du graphique ici : 
plt.xlim((-4, 37)) 
plt.ylim((26, 48))
plt.savefig('git/datatreatment/HeatMaps/heatmapRBF3.png')


########################################################################################################

#Linear kernel
linear_clf = svm.OneClassSVM(nu=0.1, kernel="linear")
linear_clf.fit(data)

xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50)) #Lat et long max de la grille
# plot the line, the points, and the nearest vectors to the plane
linear_Z = linear_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
linear_Z = linear_Z.reshape(xx.shape)

plt.title("Heat map")
plt.contourf(xx, yy, linear_Z, levels=np.linspace(linear_Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, linear_Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, linear_Z, levels=[0, linear_Z.max()], colors='palevioletred')

pt = plt.scatter(data[:, 0], data[:, 1], c='gold', s=40) #Trace les points : indiquer long et lat de chaque point
#c:color et s:forme
plt.axis('tight')
#Changer les axes du graphique ici : 
plt.xlim((-4, 37)) 
plt.ylim((26, 48))
plt.savefig('git/datatreatment/HeatMaps/heatmapLinear.png')

########################################################################################################

#Polynomial kernel
poly_clf = svm.OneClassSVM(nu=0.1, kernel="poly", degree=3, gamma="auto")
poly_clf.fit(data)

xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50)) #Lat et long max de la grille
# plot the line, the points, and the nearest vectors to the plane
poly_Z = poly_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
poly_Z = poly_Z.reshape(xx.shape)

plt.title("Heat map")
plt.contourf(xx, yy, poly_Z, levels=np.linspace(poly_Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, poly_Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, poly_Z, levels=[0, poly_Z.max()], colors='palevioletred')

pt = plt.scatter(data[:, 0], data[:, 1], c='gold', s=40) #Trace les points : indiquer long et lat de chaque point
#c:color et s:forme
plt.axis('tight')
#Changer les axes du graphique ici : 
plt.xlim((-4, 37)) 
plt.ylim((26, 48))
plt.savefig('git/datatreatment/HeatMaps/heatmapPoly.png')
