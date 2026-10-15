# NYC Taxi Demand

Пайплайн данных и почасовой прогноз спроса на такси по районам Нью-Йорка на открытых данных NYC TLC.

Стек: PySpark, PostgreSQL, LightGBM, MLflow, Docker.

## Задача

Таксопарку нужно сегодня решить, сколько машин держать завтра в каждом районе. Модель прогнозирует на сутки вперёд число поездок, начавшихся в каждой из 263 зон за каждый час.

## Результаты

Данные: поездки жёлтых такси за январь–июнь 2024, тест на июне (3,46 млн поездок). Метрика WAPE = Σ|ошибка| / Σ факт. MAPE не подходит, потому что в 56% пар «зона × час» поездок нет совсем.

| Прогноз | WAPE | MAE, поездок/час |
|---|---|---|
| Тот же час неделю назад | 0.238 | 4.35 |
| Среднее за тот же час за 4 недели | 0.185 | 3.38 |
| **LightGBM (Poisson)** | **0.177** | **3.23** |

- LightGBM на **26% точнее** сезонного наивного прогноза. От среднего за 4 недели его отделяют всего 4%: почти весь сигнал в недельной сезонности, и модель опирается на неё же (74% важности по gain).
- Точность зависит от объёма спроса: в Манхэттене WAPE 0.16, в Квинсе (аэропорты) 0.25. В Бруклине, Бронксе и на Статен-Айленде жёлтых такси мало, относительная ошибка там больше 0.6. Для этих районов нужны данные зелёных такси и такси по вызову (FHV).

## Пайплайн

1. **Загрузка** (`download.py`): 6 parquet-файлов, 20,3 млн поездок, докачка при обрыве.
2. **Контроль качества и очистка** (PySpark, `clean.py`): дата вне периода, длительность ≤ 0 или > 6 ч, неизвестная зона, отрицательный тариф (отмены), дубликаты. Удалено 2,0% строк, отчёт по каждому правилу в `reports/quality_report.json`. Поездки Flex Fare (фиксированная цена в приложении, без данных счётчика) оставлены: они тоже занимают машину.
3. **Агрегация** (PySpark → PostgreSQL по JDBC, `aggregate.py`): полная сетка 263 зоны × 4368 часов, часы без поездок заполнены нулями.
4. **Витрина признаков** (SQL, `features.sql`): лаги 24 и 168 часов, среднее за тот же час за 4 недели, средние за прошлые сутки и неделю. Все признаки сдвинуты минимум на 24 часа, чтобы не было утечки будущего.
5. **Модель** (`train.py`): LightGBM с функцией потерь Пуассона, разбиение по времени (обучение 29 января – 30 апреля, май для ранней остановки, июнь для теста). Параметры, метрики, важности и модель логируются в MLflow.

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
