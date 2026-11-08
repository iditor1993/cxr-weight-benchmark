-- ICU 住院期间的床旁便携 AP 胸片（对应 研究方案_v4 第 5.1 节）

CREATE OR REPLACE TEMP TABLE cxr AS
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.first_careunit,
    i.intime,
    i.outtime,
    m.study_id,
    m.dicom_id,
    m.study_datetime,
    m.view_position,
    m.procedure_desc
FROM main.icustays i
JOIN main.cxr_metadata m
  ON m.subject_id = i.subject_id
 AND m.study_datetime BETWEEN i.intime AND i.outtime
WHERE m.view_position = 'AP'
  AND m.procedure_desc ILIKE '%PORTABLE%';
