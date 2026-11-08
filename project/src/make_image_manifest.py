"""Stage 2a：为队列 A 生成 DICOM 清单（索引集与全集）。"""

from __future__ import annotations

import duckdb

from common import PROCESSED_DIR, load_config, setup_logger


def main() -> None:
    log = setup_logger("make_image_manifest")
    cfg = load_config()
    con = duckdb.connect(cfg["data"]["duckdb"], read_only=True)
    con.execute("PRAGMA threads=8")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE files AS
        SELECT dicom_id, zip_path, inner_path, size_bytes, view_position, procedure_desc
        FROM main.cxr_files
    """)

    for name, csv_name in (("index", "cohort_a_index.csv"),
                           ("all", "cohort_a_all_images.csv"),
                           ("cohort_b", "cohort_b_index.csv")):
        csv_path = str(PROCESSED_DIR / csv_name).replace("\\", "/")
        out = PROCESSED_DIR / f"image_manifest_{name}.csv"
        con.execute(f"""
            COPY (
                SELECT c.subject_id, c.stay_id, c.study_id, c.dicom_id,
                       c.study_datetime, c.weight_kg, c.gap_hours,
                       f.zip_path, f.inner_path, f.size_bytes,
                       f.procedure_desc
                FROM read_csv_auto('{csv_path}') c
                JOIN files f USING (dicom_id)
                ORDER BY f.zip_path, f.inner_path
            ) TO '{str(out).replace(chr(92), '/')}' (HEADER, DELIMITER ',')
        """)
        n = con.execute(f"SELECT count(*) FROM read_csv_auto('{str(out).replace(chr(92), '/')}')").fetchone()[0]
        log.info("manifest %s: %d rows -> %s", name, n, out)

    # 队列 A 影像透明度：队列 A 中有多少 dicom_id 不在 cxr_files
    missing = con.execute("""
        SELECT count(*) FROM read_csv_auto(?)
        WHERE dicom_id NOT IN (SELECT dicom_id FROM files)
    """, [str(PROCESSED_DIR / "cohort_a_all_images.csv").replace("\\", "/")]).fetchone()[0]
    log.info("cohort A images missing from cxr_files: %d", missing)


if __name__ == "__main__":
    main()
