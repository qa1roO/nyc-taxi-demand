"""
Download NYC TLC yellow taxi trip records and the taxi zone lookup.

Usage (inside the container):
	python src/download.py --start 2024-01 --end 2024-06
"""

import argparse
from pathlib import Path

import requests

BASE_URL = 'https://d37ci6vzurychx.cloudfront.net'
RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def months(start: str, end: str) -> list[str]:
	year, month = map(int, start.split('-'))
	end_year, end_month = map(int, end.split('-'))
	result = []
	while (year, month) <= (end_year, end_month):
		result.append(f'{year}-{month:02d}')
		year, month = (year + 1, 1) if month == 12 else (year, month + 1)
	return result


def download(url: str, target: Path) -> None:
	if target.exists():
		print(f'skip {target.name}: already downloaded')
		return
	tmp = target.with_suffix(target.suffix + '.part')
	with requests.get(url, stream=True, timeout=60) as response:
		response.raise_for_status()
		with open(tmp, 'wb') as file:
			for chunk in response.iter_content(chunk_size=1 << 20):
				file.write(chunk)
	tmp.rename(target)
	print(f'{target.name}: {target.stat().st_size / 2**20:.1f} MB')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', default='2024-01', help='first month, YYYY-MM')
	parser.add_argument('--end', default='2024-06', help='last month, YYYY-MM')
	args = parser.parse_args()

	trips_dir = RAW_DIR / 'yellow'
	trips_dir.mkdir(parents=True, exist_ok=True)
	for month in months(args.start, args.end):
		name = f'yellow_tripdata_{month}.parquet'
		download(f'{BASE_URL}/trip-data/{name}', trips_dir / name)
	download(f'{BASE_URL}/misc/taxi_zone_lookup.csv', RAW_DIR / 'taxi_zone_lookup.csv')


if __name__ == '__main__':
	main()
