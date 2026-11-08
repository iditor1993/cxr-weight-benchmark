# cxr-weight-benchmark

Code for the study **"Formula-based weight surrogates are unreliable in intensive care:
benchmarking against portable chest radiography deep learning"**.

The study quantifies the accuracy of the non-measurement body-weight estimates used in intensive
care (population means, height-and-sex regression, ideal-body-weight formulas) and benchmarks them
against a three-seed ResNet-50 ensemble trained on portable anteroposterior chest radiographs from
MIMIC-CXR, using bed-scale weights as the reference standard.

This repository contains **code only**. The manuscript, figures and submission files are not
published here.

## Repository layout

```
project/
  config/config.yaml   analysis configuration (paths, cohort rules, training settings)
  sql/                 MIMIC-IV cohort-extraction SQL (5 scripts)
  src/                 Python pipeline: cohort building, manifest and preprocessing,
                       training, evaluation, statistics, attribution and figures
scripts/               exploratory cohort and literature-search utilities
```

Main entry points:

| Script | Purpose |
|---|---|
| `project/src/extract_cohort.py` | build the analysis cohort from MIMIC-IV / MIMIC-CXR |
| `project/src/make_image_manifest.py` | link index radiographs to image files |
| `project/src/preprocess_images.py` | decode, crop and cache images |
| `project/src/train.py` | train a backbone (ResNet-50, Swin-T, ...) for weight regression |
| `project/src/evaluate.py` | evaluate a single model or an ensemble on a split |
| `project/src/uncertainty.py` | prediction-interval coverage and calibration |
| `project/src/dose_simulation.py` | simulated dosing impact of weight error |
| `project/src/final_audit.py` | completeness audit of result artefacts |

Statistical analyses used in the paper (subgroup analysis, formula stratification, decision-curve
analysis, temporal validation, attribution) are separate modules under `project/src/`.

## Data availability

The MIMIC-IV (v3.1) and MIMIC-CXR (v2.1.0) databases are available from PhysioNet to credentialed
users under a data use agreement:

- MIMIC-IV: https://doi.org/10.13026/6zdk-td73
- MIMIC-CXR: https://doi.org/10.13026/C2JT1Q

**No patient-level data are included in this repository**, and none can be redistributed by the
authors. Access requires CITI "Data or Specimens Only Research" training and signed data use
agreements. Model weights are not distributed either; they are derived from credentialed data.

## Installation

Python 3.11.7 and CUDA 12.8 were used for the reported experiments.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduction

Set the database paths in `project/config/config.yaml`, then run from the repository root:

```bash
python project/src/extract_cohort.py
python project/src/make_image_manifest.py
python project/src/preprocess_images.py --set index --workers 8
python project/src/train.py --backbone resnet50 --size 308 --seed 1 --tag resnet50_308_s1
# repeat for seeds 2 and 3, then evaluate the ensemble
python project/src/evaluate.py --tag resnet50_308_ensemble --backbone resnet50 --size 308 \
  --checkpoints project/data/features/checkpoints/resnet50_308_s1.pt \
                project/data/features/checkpoints/resnet50_308_s2.pt \
                project/data/features/checkpoints/resnet50_308_s3.pt
python project/src/uncertainty.py --tag resnet50_308_ensemble
python project/src/dose_simulation.py --tag resnet50_308_ensemble
python project/src/final_audit.py
```

## License

MIT License (see `LICENSE`).

## Contact

Xuming Pan, Department of Emergency, The Second Affiliated Hospital of Zhejiang Chinese Medical
University, Hangzhou, Zhejiang, China — 20194112@zcmu.edu.cn
