-- 队列 B（敏感性）：每张影像取 ±24h 内最近的任意体重来源（224639/226512/226531）

CREATE OR REPLACE TEMP TABLE pairs_any AS
SELECT * FROM (
    SELECT
        c.subject_id,
        c.hadm_id,
        c.stay_id,
        c.first_careunit,
        c.study_id,
        c.dicom_id,
        c.study_datetime,
        w.charttime  AS weight_charttime,
        w.weight_kg,
        w.itemid     AS weight_itemid,
        abs(epoch(c.study_datetime - w.charttime)) / 3600.0 AS gap_hours,
        row_number() OVER (
            PARTITION BY c.dicom_id
            ORDER BY abs(epoch(c.study_datetime - w.charttime)) ASC,
                     w.charttime ASC,
                     w.itemid ASC
        ) AS w_rank
    FROM cxr c
    JOIN w_clean w
      ON w.stay_id = c.stay_id
     AND w.in_range
     AND w.charttime BETWEEN c.study_datetime - INTERVAL 24 HOUR
                         AND c.study_datetime + INTERVAL 24 HOUR
) t
WHERE w_rank = 1;
