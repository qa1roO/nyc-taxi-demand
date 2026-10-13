"""
Step 2: data quality checks and cleaning.

Applies the drop rules to the raw trips, writes the clean trips as parquet
partitioned by month and saves a report with how many rows each rule caught.

Kept on purpose:
	- Flex Fare trips (payment_type = 0): upfront-priced app trips whose meter fields
	  (passenger_count, RatecodeID, surcharges) are NULL. They are real trips and occupy a cab.
	- zero distance trips: the cab was still occupied.

Usage (inside the container):
	python src/clean.py --start 2024-01 --end 2024-06
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
CLEAN_DIR = ROOT / 'data' / 'clean' / 'trips'
REPORT_PATH = ROOT / 'reports' / 'quality_report.json'

UNKNOWN_ZONES = [264, 265]
MAX_DURATION_MIN = 6 * 60


def month_bounds(start: str, end: str) -> tuple[datetime, datetime]:
	first = datetime.strptime(start, '%Y-%m')
	last = datetime.strptime(end, '%Y-%m')
	after_last = last.replace(year=last.year + last.month // 12, month=last.month % 12 + 1)
	return first, after_last


def ntz(moment: datetime):
	"""Literal of the same type as the TLC timestamps (no time zone), so no zone shift sneaks in."""
	return F.lit(moment.strftime('%Y-%m-%d %H:%M:%S')).cast('timestamp_ntz')


def drop_rules(first: datetime, after_last: datetime) -> dict:
	"""Each rule is a boolean column: True means the row is dropped."""
	pickup = F.col('pickup_datetime')
	duration_min = (F.unix_timestamp('dropoff_datetime') - F.unix_timestamp('pickup_datetime')) / 60
	return {
		'pickup_outside_period': (pickup < ntz(first)) | (pickup >= ntz(after_last)),
		'non_positive_duration': duration_min <= 0,
		'duration_over_6h': duration_min > MAX_DURATION_MIN,
		'unknown_pickup_zone': F.col('pickup_zone').isin(UNKNOWN_ZONES),
		'negative_fare_reversal': F.col('fare_amount') < 0,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', default='2024-01', help='first month, YYYY-MM')
	parser.add_argument('--end', default='2024-06', help='last month, YYYY-MM')
	args = parser.parse_args()
	first, after_last = month_bounds(args.start, args.end)

	spark = (
		SparkSession.builder
		.appName('nyc-taxi-clean')
		.master('local[*]')
		.config('spark.driver.memory', '4g')
		.config('spark.ui.showConsoleProgress', 'false')
		.getOrCreate()
	)
	spark.sparkContext.setLogLevel('ERROR')

	raw = spark.read.parquet(str(RAW_DIR / 'yellow'))
	trips = raw.select(
		F.col('VendorID').alias('vendor_id'),
		F.col('tpep_pickup_datetime').alias('pickup_datetime'),
		F.col('tpep_dropoff_datetime').alias('dropoff_datetime'),
		F.col('PULocationID').alias('pickup_zone'),
		F.col('DOLocationID').alias('dropoff_zone'),
		'passenger_count',
		'trip_distance',
		'fare_amount',
		'total_amount',
		'payment_type',
	)
	total = trips.count()

	rules = drop_rules(first, after_last)
	# Rules overlap, so count each one on its own and the union separately
	caught = trips.agg(*[F.sum(rule.cast('int')).alias(name) for name, rule in rules.items()]).first().asDict()

	any_rule = F.lit(False)
	for rule in rules.values():
		any_rule = any_rule | F.coalesce(rule, F.lit(False))
	passed = trips.filter(~any_rule)
	after_rules = passed.count()

	clean = (
		passed.dropDuplicates()
		.withColumn('is_flex_fare', F.col('payment_type') == 0)
		.withColumn('month', F.date_format('pickup_datetime', 'yyyy-MM'))
	)
	clean.write.mode('overwrite').partitionBy('month').parquet(str(CLEAN_DIR))

	written = spark.read.parquet(str(CLEAN_DIR))
	kept = written.count()
	flex = written.filter('is_flex_fare').count()
	spark.stop()

	report = {
		'period': f'{args.start}..{args.end}',
		'raw_rows': total,
		'caught_by_rule': caught,
		'exact_duplicates': after_rules - kept,
		'dropped_rows': total - kept,
		'dropped_share': round((total - kept) / total, 5),
		'clean_rows': kept,
		'kept_flex_fare_rows': flex,
	}
	REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
	REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')

	print(f'Raw rows:   {total:>12,}')
	for name, count in caught.items():
		print(f'  {name:<26} {count:>10,}  {count / total:7.2%}')
	print(f'  {"exact_duplicates":<26} {after_rules - kept:>10,}')
	print(f'Dropped:    {total - kept:>12,}  {(total - kept) / total:7.2%}  (a row can break several rules)')
	print(f'Clean rows: {kept:>12,}  of them Flex Fare: {flex:,}')
	print(f'Saved to {CLEAN_DIR.relative_to(ROOT)}, report in {REPORT_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
	main()
