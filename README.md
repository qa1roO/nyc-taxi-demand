# NYC Taxi Demand

Пайплайн данных и почасовой прогноз спроса на такси по районам Нью-Йорка на открытых данных NYC TLC.

Стек: PySpark, PostgreSQL, LightGBM, MLflow, Docker.

## Запуск

Нужен только Docker Desktop: Java, Spark, Python и PostgreSQL работают в контейнерах.

```bash
docker compose build                                               # собрать образ: Python + Java + PySpark
docker compose up -d postgres                                      # поднять PostgreSQL в фоне
docker compose run --rm app python src/download.py --start 2024-01 --end 2024-06
docker compose run --rm app python src/explore.py
docker compose run --rm app python src/clean.py --start 2024-01 --end 2024-06
docker compose run --rm app python src/aggregate.py --start 2024-01 --end 2024-06
docker compose exec postgres psql -U taxi -d taxi -f /sql/features.sql
docker compose run --rm app python src/train.py
```

Эксперименты MLflow (http://127.0.0.1:5000):

```bash
docker compose run --rm --service-ports app mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
```

JupyterLab для исследования данных:

```bash
docker compose run --rm --service-ports app jupyter lab --ip 0.0.0.0 --allow-root --no-browser
```

Ссылку с токеном из вывода откройте в браузере (http://127.0.0.1:8888/...).

## Структура

```text
.
├── data/raw/            # сырые данные (не хранятся в git)
├── notebooks/           # исследование
├── src/
│   ├── download.py      # загрузка поездок и справочника зон
│   ├── explore.py       # первый взгляд на данные: схема, объём, нарушения правил качества
│   ├── clean.py         # очистка по правилам качества, отчёт reports/quality_report.json
│   ├── aggregate.py     # спрос зона × час (с нулями) в PostgreSQL
│   └── train.py         # LightGBM против простых прогнозов, метрика WAPE, MLflow
├── sql/
│   └── features.sql     # витрина признаков: лаги и скользящие средние оконными функциями
├── Dockerfile           # Python + Java + PySpark + JDBC-драйвер PostgreSQL
└── docker-compose.yml   # контейнер с кодом и PostgreSQL
```

## Данные

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page): поездки жёлтых такси, по одному parquet-файлу на месяц (~3 млн поездок в месяц), и справочник 265 районов (`taxi_zone_lookup.csv`).
