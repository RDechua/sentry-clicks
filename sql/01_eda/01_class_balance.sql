-- Class balance: overall positive rate, then by day, then by hour-of-day.
-- Note: TalkingData's label semantics are inverted from "fraud" — is_attributed=0
-- means the click did not convert (likely fraud, majority class). We report
-- mean(is_attributed) = positive_rate; fraud rate = 1 - positive_rate.
-- `level` distinguishes the aggregation grain in the output.

COPY (
    SELECT
        'overall'                            AS level,
        NULL::DATE                           AS day,
        NULL::INTEGER                        AS hour_of_day,
        COUNT(*)                             AS n_clicks,
        SUM(is_attributed)                   AS n_attributed,
        AVG(is_attributed::DOUBLE)           AS positive_rate
    FROM clicks

    UNION ALL

    SELECT
        'by_day',
        DATE_TRUNC('day', click_time)::DATE,
        NULL::INTEGER,
        COUNT(*),
        SUM(is_attributed),
        AVG(is_attributed::DOUBLE)
    FROM clicks
    GROUP BY DATE_TRUNC('day', click_time)

    UNION ALL

    SELECT
        'by_hour_of_day',
        NULL::DATE,
        EXTRACT(hour FROM click_time)::INTEGER,
        COUNT(*),
        SUM(is_attributed),
        AVG(is_attributed::DOUBLE)
    FROM clicks
    GROUP BY EXTRACT(hour FROM click_time)

    ORDER BY level, day NULLS FIRST, hour_of_day NULLS FIRST
)
TO 'reports/eda/01_class_balance.csv' (HEADER, FORMAT CSV);
