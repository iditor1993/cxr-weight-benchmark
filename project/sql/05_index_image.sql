-- 索引影像：每次 ICU 住院取最早的合格配对影像
-- 对应 研究方案_v4 第 5.4 节（含回退规则：本表在“最早影像不合格”时由后续质控步骤回退）

CREATE OR REPLACE TEMP TABLE index_daily AS
SELECT * FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY stay_id
            ORDER BY study_datetime ASC, dicom_id ASC
        ) AS img_rank
    FROM pairs_daily
) t
WHERE img_rank = 1;

CREATE OR REPLACE TEMP TABLE index_any AS
SELECT * FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY stay_id
            ORDER BY study_datetime ASC, dicom_id ASC
        ) AS img_rank
    FROM pairs_any
) t
WHERE img_rank = 1;
