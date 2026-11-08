-- 体重事件清洗（对应 研究方案_v4 第 5、6 节）
-- 224639: Daily Weight (kg, 床秤)
-- 226512: Admission Weight (kg)
-- 226531: Admission Weight (lbs) -> kg

CREATE OR REPLACE TEMP TABLE w_raw AS
SELECT stay_id, subject_id, charttime, itemid, valuenum, valueuom
FROM main.chartevents
WHERE itemid IN (224639, 226512, 226531);

CREATE OR REPLACE TEMP TABLE w_clean AS
SELECT
    stay_id,
    subject_id,
    charttime,
    itemid,
    CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END AS weight_kg,
    CASE WHEN itemid = 226531 THEN valuenum / 2.2046226218 ELSE valuenum END
        BETWEEN 20.0 AND 350.0 AS in_range
FROM w_raw
WHERE valuenum IS NOT NULL;
