"""
Step 1: first look at the raw trips with PySpark.

Prints the schema, row counts per month and how many rows break basic sanity rules.
These rules become the data quality checks of the pipeline in step 2.

Usage (inside the container):
	python src/explore.py
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def main():
	spark = (
		SparkSession.builder
		.appName('nyc-taxi-explore')
		.master('local[*]')
		.config('spark.driver.memory', '4g')
		.config('spark.ui.showConsoleProgress', 'false')
		.getOrCreate()
	)
	spark.sparkContext.setLogLevel('WARN')

	trips = spark.read.parquet(str(RAW_DIR / 'yellow'))
	trips.printSchema()

	total = trips.count()
	print(f'Rows: {total:,}')

	print('Rows per pickup month (note the stray dates outside the downloaded range):')
	(
		trips
		.groupBy(F.date_format('tpep_pickup_datetime', 'yyyy-MM').alias('month'))
		.count()
		.orderBy('month')
		.show(50)
	)

	duration_min = (
		F.unix_timestamp('tpep_dropoff_datetime') - F.unix_timestamp('tpep_pickup_datetime')
	) / 60

	# Each rule is a boolean column: True means the row looks wrong
	rules = {
		'non_positive_duration': duration_min <= 0,
		'longer_than_6h': duration_min > 6 * 60,
		'zero_distance': F.col('trip_distance') <= 0,
		'negative_fare': F.col('fare_amount') < 0,
		'no_passengers': F.col('passenger_count') == 0,
		'null_passengers': F.col('passenger_count').isNull(),
		'unknown_zone': F.col('PULocationID').isin(264, 265),
	}
	counts = trips.agg(*[F.sum(rule.cast('int')).alias(name) for name, rule in rules.items()]).first()

	print('Suspicious rows by rule:')
	for name in rules:
		print(f'  {name:<22} {counts[name]:>12,}  {counts[name] / total:7.2%}')

	duplicates = total - trips.dropDuplicates().count()
	print(f'  {"exact_duplicates":<22} {duplicates:>12,}  {duplicates / total:7.2%}')

	spark.stop()


if __name__ == '__main__':
	main()
