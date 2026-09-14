-- ============================================================================
-- queries.sql
--
-- SQL versions of the core business questions for the United Airlines
-- Flight Delay & Operational Performance Analysis project.
--
-- Run against sql/united_flights.db (built by sql/load_to_sqlite.py), e.g.:
--   sqlite3 sql/united_flights.db < sql/queries.sql
-- or paste individual queries into any SQL client / pandas.read_sql().
--
-- Table: flights
-- Key columns: FL_DATE, ORIGIN, DEST, ARR_DELAY, DEP_DELAY, CANCELLED,
-- DIVERTED, YEAR, MONTH, DAY_OF_WEEK, IS_DELAYED_15,
-- DELAY_DUE_CARRIER / WEATHER / NAS / SECURITY / LATE_AIRCRAFT
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1: Which origin airports have the worst average arrival delay?
-- (Restricted to airports with at least 200 flights so small airports with
--  one bad flight don't skew the ranking.)
-- ----------------------------------------------------------------------------
SELECT
    ORIGIN,
    COUNT(*)                    AS num_flights,
    ROUND(AVG(ARR_DELAY), 1)    AS avg_arrival_delay_min
FROM flights
WHERE CANCELLED = 0 AND DIVERTED = 0
GROUP BY ORIGIN
HAVING COUNT(*) >= 200
ORDER BY avg_arrival_delay_min DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- Q2: Which months have the worst average arrival delay?
-- ----------------------------------------------------------------------------
SELECT
    MONTH,
    COUNT(*)                    AS num_flights,
    ROUND(AVG(ARR_DELAY), 1)    AS avg_arrival_delay_min
FROM flights
WHERE CANCELLED = 0 AND DIVERTED = 0
GROUP BY MONTH
ORDER BY MONTH;


-- ----------------------------------------------------------------------------
-- Q3: Total delay-minutes by cause, and as a % of the total.
-- ----------------------------------------------------------------------------
WITH cause_totals AS (
    SELECT
        SUM(DELAY_DUE_CARRIER)      AS carrier_min,
        SUM(DELAY_DUE_WEATHER)      AS weather_min,
        SUM(DELAY_DUE_NAS)          AS nas_min,
        SUM(DELAY_DUE_SECURITY)     AS security_min,
        SUM(DELAY_DUE_LATE_AIRCRAFT) AS late_aircraft_min
    FROM flights
    WHERE CANCELLED = 0 AND DIVERTED = 0
)
SELECT
    'CARRIER'        AS delay_cause, carrier_min        AS total_minutes FROM cause_totals
UNION ALL
SELECT 'WEATHER',        weather_min        FROM cause_totals
UNION ALL
SELECT 'NAS',             nas_min            FROM cause_totals
UNION ALL
SELECT 'SECURITY',        security_min       FROM cause_totals
UNION ALL
SELECT 'LATE_AIRCRAFT',   late_aircraft_min  FROM cause_totals
ORDER BY total_minutes DESC;


-- ----------------------------------------------------------------------------
-- Q4: Is on-time performance improving or worsening year over year?
-- (% of flights delayed 15+ minutes on arrival, by year)
-- ----------------------------------------------------------------------------
SELECT
    YEAR,
    COUNT(*)                                              AS num_flights,
    ROUND(AVG(ARR_DELAY), 1)                              AS avg_arrival_delay_min,
    ROUND(100.0 * SUM(IS_DELAYED_15) / COUNT(*), 1)       AS pct_delayed_15plus
FROM flights
WHERE CANCELLED = 0 AND DIVERTED = 0
GROUP BY YEAR
ORDER BY YEAR;


-- ----------------------------------------------------------------------------
-- Bonus: Overall cancellation rate, and cancellation rate by origin airport
-- (top 10 worst, min 200 scheduled flights).
-- ----------------------------------------------------------------------------
SELECT
    ORIGIN,
    COUNT(*)                                        AS num_flights,
    ROUND(100.0 * SUM(CANCELLED) / COUNT(*), 2)     AS cancellation_rate_pct
FROM flights
GROUP BY ORIGIN
HAVING COUNT(*) >= 200
ORDER BY cancellation_rate_pct DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- Bonus: Worst routes (origin-destination pairs) by average arrival delay,
-- minimum 100 flights on the route.
-- ----------------------------------------------------------------------------
SELECT
    ORIGIN,
    DEST,
    COUNT(*)                    AS num_flights,
    ROUND(AVG(ARR_DELAY), 1)    AS avg_arrival_delay_min
FROM flights
WHERE CANCELLED = 0 AND DIVERTED = 0
GROUP BY ORIGIN, DEST
HAVING COUNT(*) >= 100
ORDER BY avg_arrival_delay_min DESC
LIMIT 10;
