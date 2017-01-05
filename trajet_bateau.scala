val mmsi ="247145870"
val inputRDD = sc.textFile("/home/ana/scrapper_pierre/aisdata/AIS_dump_for_sparkV2.txt")
val ais = inputRDD.map(_.split("\t"))
val ship = ais.filter(line => line.contains(mmsi)).map(col => ((col(0), col(1), col(3)), col(7) ))
val locations = ship.reduceByKey((key, value) => key) //supprime les doublons si même location et même vitesse