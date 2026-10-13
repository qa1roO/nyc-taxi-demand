-- Step 3b: feature mart for the demand model, built with window functions.
--
-- Forecast horizon is one day ahead: tomorrow's cabs are planned today, so every
-- feature only uses demand from at least 24 hours before the target hour.
-- The window is ordered by hour inside each zone. Because demand_hourly has a row
-- for every zone and every hour, "N rows back" is exactly "N hours back".
--
-- Usage:
--   docker compose exec postgres psql -U taxi -d taxi -f /sql/features.sql

\set ON_ERROR_STOP on

DROP TABLE IF EXISTS demand_features;

CREATE TABLE demand_features AS
SELECT
	d.zone_id,
	z.borough,
	d.hour,
	d.trips,
	EXTRACT(HOUR FROM d.hour)::int AS hour_of_day,
	EXTRACT(ISODOW FROM d.hour)::int AS day_of_week,  -- 1 = Monday ... 7 = Sunday
	-- same hour yesterday and a week ago; lag_168h is also the seasonal naive baseline
	LAG(d.trips, 24) OVER w AS lag_24h,
	LAG(d.trips, 168) OVER w AS lag_168h,
	-- same hour on the same weekday, averaged over the last 4 weeks
	round((LAG(d.trips, 168) OVER w + LAG(d.trips, 336) OVER w
		+ LAG(d.trips, 504) OVER w + LAG(d.trips, 672) OVER w) / 4.0, 3) AS same_hour_4w_mean,
	-- the whole previous day (24 hours ending 24 hours before the target)
	round(AVG(d.trips) OVER (w ROWS BETWEEN 47 PRECEDING AND 24 PRECEDING), 3) AS prev_day_mean,
	-- the previous week, ending 24 hours before the target
	round(AVG(d.trips) OVER (w ROWS BETWEEN 191 PRECEDING AND 24 PRECEDING), 3) AS prev_week_mean
FROM demand_hourly d
JOIN zones z USING (zone_id)
WINDOW w AS (PARTITION BY d.zone_id ORDER BY d.hour);

-- The first 4 weeks have no full history yet, so their lags are NULL
DELETE FROM demand_features WHERE same_hour_4w_mean IS NULL;

ALTER TABLE demand_features ADD PRIMARY KEY (zone_id, hour);
CREATE INDEX ON demand_features (hour);

SELECT
	count(*) AS rows,
	count(DISTINCT zone_id) AS zones,
	min(hour) AS first_hour,
	max(hour) AS last_hour,
	round(avg(trips), 2) AS mean_trips
FROM demand_features;
