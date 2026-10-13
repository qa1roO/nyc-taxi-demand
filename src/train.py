"""
Step 4: day-ahead demand model, compared with simple baselines and tracked in MLflow.

Reads the feature mart demand_features from PostgreSQL and splits it by time:
the last month is the test set, the month before it is used for early stopping,
everything earlier is for training. A random split would leak the future into training.

The model has to beat two forecasts that need no ML at all:
	lag_168h           - the same hour a week ago (seasonal naive)
	same_hour_4w_mean  - the same hour and weekday averaged over the last 4 weeks

Metric: WAPE = sum(|error|) / sum(actual). MAPE is undefined here because many
zone-hours have 0 trips.

Usage (inside the container, after sql/features.sql):
	python src/train.py
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0   # then open http://127.0.0.1:5000
"""

import argparse
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]

CATEGORICAL = ['zone_id', 'borough']
FEATURES = CATEGORICAL + [
	'hour_of_day', 'day_of_week',
	'lag_24h', 'lag_168h', 'same_hour_4w_mean', 'prev_day_mean', 'prev_week_mean',
]
BASELINES = ['lag_168h', 'same_hour_4w_mean']
PARAMS = {
	'objective': 'poisson',  # demand is a count: non-negative, often 0
	'learning_rate': 0.05,
	'num_leaves': 63,
	'min_child_samples': 50,
	'subsample': 0.8,
	'subsample_freq': 1,
	'colsample_bytree': 0.8,
	'n_estimators': 3000,
	'random_state': 42,
	'verbose': -1,
}


def wape(actual, predicted) -> float:
	return float(np.abs(actual - predicted).sum() / actual.sum())


def mae(actual, predicted) -> float:
	return float(np.abs(actual - predicted).mean())


def load_features() -> pd.DataFrame:
	# POSTGRES_URL is the JDBC url Spark uses: jdbc:postgresql://host:port/db
	address = os.environ['POSTGRES_URL'].removeprefix('jdbc:postgresql://')
	user, password = os.environ['POSTGRES_USER'], os.environ['POSTGRES_PASSWORD']
	engine = create_engine(f'postgresql+psycopg://{user}:{password}@{address}')
	df = pd.read_sql('SELECT * FROM demand_features ORDER BY hour, zone_id', engine)
	for column in CATEGORICAL:
		df[column] = df[column].astype('category')
	return df


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--valid-month', default='2024-05', help='month for early stopping, YYYY-MM')
	parser.add_argument('--test-month', default='2024-06', help='month for the final score, YYYY-MM')
	args = parser.parse_args()

	df = load_features()
	month = df['hour'].dt.strftime('%Y-%m')
	train = df[month < args.valid_month]
	valid = df[month == args.valid_month]
	test = df[month == args.test_month]
	print(f'Rows: train {len(train):,}, valid {len(valid):,}, test {len(test):,}')

	mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', f'sqlite:///{ROOT / "mlflow.db"}'))
	mlflow.set_experiment('nyc-taxi-demand')

	with mlflow.start_run(run_name='lightgbm-day-ahead'):
		mlflow.log_params({**PARAMS, 'features': ','.join(FEATURES), 'valid_month': args.valid_month, 'test_month': args.test_month})

		model = lgb.LGBMRegressor(**PARAMS)
		model.fit(
			train[FEATURES], train['trips'],
			eval_set=[(valid[FEATURES], valid['trips'])],
			callbacks=[lgb.early_stopping(100, verbose=False)],
		)
		mlflow.log_param('best_iteration', model.best_iteration_)

		predictions = {name: test[name] for name in BASELINES}
		predictions['lightgbm'] = model.predict(test[FEATURES])

		results = pd.DataFrame({
			name: {'wape': wape(test['trips'], pred), 'mae': mae(test['trips'], pred)}
			for name, pred in predictions.items()
		}).T
		for name, row in results.iterrows():
			mlflow.log_metrics({f'test_wape_{name}': row['wape'], f'test_mae_{name}': row['mae']})

		# Where the model is good or bad: WAPE per borough
		by_borough = (
			test.assign(error=np.abs(test['trips'] - predictions['lightgbm']))
			.groupby('borough', observed=True)[['error', 'trips']].sum()
			.query('trips > 0')
		)
		by_borough['wape'] = by_borough['error'] / by_borough['trips']
		mlflow.log_table(by_borough.reset_index(), 'wape_by_borough.json')

		importance = pd.Series(model.booster_.feature_importance('gain'), index=FEATURES).sort_values(ascending=False)
		mlflow.log_dict((importance / importance.sum()).round(4).to_dict(), 'feature_importance.json')
		mlflow.lightgbm.log_model(model, name='model')

	print(f'Best iteration: {model.best_iteration_}')
	print(f'\nTest month {args.test_month}, {int(test["trips"].sum()):,} trips:')
	print(results.round(3).to_string())
	gain = 1 - results.loc['lightgbm', 'wape'] / results.loc['lag_168h', 'wape']
	print(f'\nLightGBM vs same hour a week ago: WAPE {gain:+.1%} better')
	print('\nWAPE by borough (LightGBM):')
	print(by_borough['wape'].round(3).sort_values().to_string())
	print('\nFeature importance (share of gain):')
	print((importance / importance.sum()).round(3).to_string())


if __name__ == '__main__':
	main()
