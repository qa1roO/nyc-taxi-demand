from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.master('local[*]').config('spark.driver.memory', '4g').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
trips = spark.read.parquet('data/raw/yellow').withColumn(
    'group', F.when(F.col('passenger_count').isNull(), 'null_passengers').otherwise('normal'))

for column in ['payment_type', 'RatecodeID', 'VendorID']:
    trips.groupBy('group', column).count().orderBy('group', F.desc('count')).show(20)

trips.groupBy('group').agg(
    F.count('*').alias('trips'),
    F.expr('percentile_approx(fare_amount, 0.5)').alias('median_fare'),
    F.expr('percentile_approx(trip_distance, 0.5)').alias('median_distance'),
    F.avg(F.col('congestion_surcharge').isNull().cast('int')).alias('share_null_congestion'),
).show()