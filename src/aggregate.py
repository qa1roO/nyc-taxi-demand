"""
Step 3: hourly demand per pickup zone, loaded into PostgreSQL.

Demand is the number of trips that started in a zone during an hour. Counting
with groupBy alone leaves out the zone-hours with no trips, so the counts are
joined onto a full grid of zones x hours and the gaps are filled with 0. Without
the zeros the model never sees a quiet hour, and SQL window functions such as
LAG would look at the previous row instead of the previous hour.

Writes two tables over JDBC:
	zones          - the TLC zone lookup (id, borough, name)
	demand_hourly  - zone_id, hour, trips for every zone and every hour of the period

Usage (inside the container, PostgreSQL must be up):
	python src/aggregate.py --start 2024-01 --end 2024-06
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from clean import UNKNOWN_ZONES, month_bounds, ntz

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / 'data' / 'clean' / 'trips'
ZONES_PATH = ROOT / 'data' / 'raw' / 'taxi_zone_lookup.csv'


def write_table(df, table: str) -> None:
	(
		df.write
		.format('jdbc')
		.option('url', os.environ['POSTGRES_URL'])
		.option('user', os.environ['POSTGRES_USER'])
		.option('password', os.environ['POSTGRES_PASSWORD'])
		.option('driver', 'org.postgresql.Driver')
		.option('dbtable', table)
		.option('batchsize', 10_000)
		.mode('overwrite')
		.save()
	)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', default='2024-01', help='first month, YYYY-MM')
	parser.add_argument('--end', default='2024-06', help='last month, YYYY-MM')
	args = parser.parse_args()
	first, after_last = month_bounds(args.start, args.end)

	spark = (
		SparkSession.builder
		.appName('nyc-taxi-aggregate')
		.master('local[*]')
		.config('spark.driver.memory', '4g')
		.config('spark.ui.showConsoleProgress', 'false')
		# PostgreSQL JDBC driver, downloaded into the image by the Dockerfile
		.config('spark.jars', os.environ['POSTGRES_JDBC_JAR'])
		.getOrCreate()
	)
	spark.sparkContext.setLogLevel('ERROR')

	zones = (
		spark.read.csv(str(ZONES_PATH), header=True, inferSchema=True)
		.select(
			F.col('LocationID').cast('int').alias('zone_id'),
			F.col('Borough').alias('borough'),
			F.col('Zone').alias('zone_name'),
		)
		.filter(~F.col('zone_id').isin(UNKNOWN_ZONES))
	)

	counts = (
		spark.read.parquet(str(CLEAN_DIR))
		.groupBy(
			F.col('pickup_zone').cast('int').alias('zone_id'),
			F.date_trunc('hour', 'pickup_datetime').alias('hour'),
		)
		.count()
		.withColumnRenamed('count', 'trips')
	)

	# Every hour of the period, end exclusive
	hours = spark.range(1).select(
		F.explode(F.sequence(ntz(first), ntz(after_last), F.expr('interval 1 hour'))).alias('hour')
	).filter(F.col('hour') < ntz(after_last))

	demand = (
		zones.select('zone_id')
		.crossJoin(hours)
		.join(counts, ['zone_id', 'hour'], 'left')
		.fillna(0, subset=['trips'])
		.withColumn('trips', F.col('trips').cast('int'))
		.cache()
	)

	write_table(zones, 'zones')
	write_table(demand, 'demand_hourly')

	n_zones, n_hours = zones.count(), hours.count()
	totals = demand.agg(F.count('*').alias('rows'), F.sum('trips').alias('trips'), F.sum((F.col('trips') == 0).cast('int')).alias('empty')).first()
	spark.stop()

	print(f'Zones x hours: {n_zones} x {n_hours:,} = {totals["rows"]:,} rows')
	print(f'Trips counted: {totals["trips"]:,}')
	print(f'Zone-hours with no trips: {totals["empty"]:,}  {totals["empty"] / totals["rows"]:.1%}')
	print('Written to PostgreSQL tables zones and demand_hourly')


if __name__ == '__main__':
	main()
