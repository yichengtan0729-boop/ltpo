# FLOC-Rad

FLOC-Rad is an image-grounded radiology report correction pipeline for evaluating and improving generalization to unseen combinations of clinical edits.

## Task

Input: a chest radiograph and an imperfect report.

Output: a structured edit set and a corrected report.

The first implementation supports `ADD_FINDING`, `REMOVE_FINDING`, `SET_NEGATION`, `SET_LATERALITY`, `SET_LOCATION`, and `SET_SEVERITY`. Train/test splits hold out complete edit combinations while retaining their atomic edits in training.

## Installation

```bash
cd floc_rad
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Fast smoke test

This creates a synthetic image-report dataset, builds seen/unseen composition splits, trains a tiny model, and writes evaluation metrics.

```bash
bash scripts/run_synthetic_smoke.sh
```

Expected output:

```text
floc_rad/outputs/synthetic/evaluation/metrics.json
```

## Dataset preparation

### OpenI / IU X-Ray

OpenI can be downloaded automatically:

```bash
python -m flocrad download --dataset openi --output-dir data/openi
python -m flocrad manifest --dataset openi --data-dir data/openi --output-csv data/openi/manifest.csv
```

### MIMIC-CXR

MIMIC-CXR requires PhysioNet credentialed access. Do not send MIMIC reports to an unapproved external API.

```bash
export PHYSIONET_USERNAME=YOUR_USERNAME
export PHYSIONET_PASSWORD=YOUR_PASSWORD
python -m flocrad download \
  --dataset mimic \
  --output-dir data/mimic \
  --max-images 10000
python -m flocrad manifest \
  --dataset mimic \
  --data-dir data/mimic \
  --output-csv data/mimic/manifest.csv
```

Set `--max-images 0` for all images. The downloader uses PhysioNet `wget` authentication and therefore requires `wget` on the system.

### CheXpert Plus

CheXpert Plus must be exported after accepting its Redivis terms. Stage the export and let the schema-aware manifest builder locate the report/image mapping table:

```bash
python -m flocrad download \
  --dataset chexpert_plus \
  --source-dir /path/to/chexpert_plus_export \
  --output-dir data/chexpert_plus
python -m flocrad manifest \
  --dataset chexpert_plus \
  --data-dir data/chexpert_plus \
  --output-csv data/chexpert_plus/manifest.csv
```

For an unsupported export schema, create a CSV with:

```text
study_id,patient_id,image_path,report,split,dataset
```

and run:

```bash
python -m flocrad manifest \
  --dataset universal \
  --data-dir /dataset/root \
  --input-csv /path/to/manifest.csv \
  --output-csv data/custom/manifest.csv
```

## End-to-end run

```bash
python -m flocrad pipeline \
  --manifest data/mimic/manifest.csv \
  --work-dir outputs/mimic \
  --device cuda \
  --image-backbone resnet18 \
  --image-pretrained true \
  --epochs 10 \
  --batch-size 32
```

The pipeline performs:

1. deterministic clinical-state parsing;
2. controlled error injection;
3. canonical edit annotation;
4. held-out composition split construction;
5. multimodal edit-predictor training;
6. seen/unseen evaluation.

## Separate commands

```bash
python -m flocrad prepare --manifest data/openi/manifest.csv --output-dir outputs/openi/splits
python -m flocrad train --data-dir outputs/openi/splits --output-dir outputs/openi/model --device cuda
python -m flocrad evaluate \
  --checkpoint outputs/openi/model/best.pt \
  --data-dir outputs/openi/splits \
  --output-dir outputs/openi/evaluation \
  --device cuda
```

## Correct one report

```bash
python -m flocrad predict \
  --checkpoint outputs/mimic/model/best.pt \
  --image /path/to/image.jpg \
  --report "Moderate right pleural effusion." \
  --device cuda
```

## Main metrics

- operator F0.5 and F1;
- exact composition fix rate;
- seen-unseen composition gap;
- non-target preservation;
- clean-report false edit rate;
- results separated by error count and operator type.

## Notes

- The deterministic parser is a reproducible baseline, not a claim of perfect clinical extraction. Replace it with RadGraph or a validated local extractor for full experiments, while keeping the canonical state and edit interfaces unchanged.
- OpenI and CheXpert Plus schemas can change. The manifest command validates required fields and fails with an actionable message rather than silently producing an empty dataset.
- `tiny` is only for smoke tests. Use `resnet18` or `resnet50` for formal experiments.
