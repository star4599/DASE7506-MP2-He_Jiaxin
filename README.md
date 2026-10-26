# DASE7506 MP2 — Offline Candidate Selection

**Author:** He Jiaxin  
**Student ID:** 3036748919  
**Repository:** [star4599/DASE7506-MP2-He_Jiaxin](https://github.com/star4599/DASE7506-MP2-He_Jiaxin)  
**Protocol:** `dase7506-mp2-open-hard-verifier-v5` | **Device:** CPU, four threads

## 1. Experimental Results

The final method combines a projected candidate ranker with arithmetic evidence and correction heads trained on out-of-fold predictions. It selects one supplied candidate offline, using no additional pretrained weights or external training datasets.

| Model | Validation accuracy | Test accuracy | Full validation time (s) | Assets (KiB) |
| :--- | ---: | ---: | ---: | ---: |
| Initial learned baseline | 91.35% | 79.87% | 0.544 | 13.3 |
| Projected ranker | 93.98% | 81.58% | 0.687 | 281.5 |
| **Final evidence ranker, LR 0.003** | **93.73%** | **84.37%** | **1.015** | **284.5** |

The final checkpoint achieves **748/798** on validation, **305/355** on validation disagreement questions, and **394/467** on public test. These are reproduction scores on the supplied bank, not an independent blind-test estimate; the final method does not improve validation accuracy over the projected ranker.

Ablations at LR 0.003 reduce validation disagreement accuracy from **85.92%** to **85.07%** without correction heads, **84.79%** without arithmetic evidence, and **85.63%** without ranking loss. Removing KL regularization or using one head seed leaves the validation predictions unchanged. Full records are in `records/report-focused/`.

Local hardware: **Intel Core i7-12700H, Windows 11**. Final training took **20.22 s**; inference assets total **291,329 bytes (0.278 MiB)**. Validation timings include fresh prediction and independent scoring processes. Full test process timing and the reference Xeon Platinum 8457C have not been measured; these local results do not certify the official 30 s limit.

## 2. Installation & Training

Use **Python 3.12**. The following commands are for **Windows CMD / Anaconda Prompt**:

```cmd
git clone https://github.com/star4599/DASE7506-MP2-He_Jiaxin.git
cd DASE7506-MP2-He_Jiaxin
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r code/requirements.txt
python verify_submission.py
cd code
python check_data.py

set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
```

Inference requires only **NumPy 2.5.3**. To retrain on CPU, install the additional dependencies and use a new output directory:

```cmd
python -m pip install -r requirements-training.txt
python train.py --config train_config.json --output runs/retrained_final
```

Training uses only train inputs/references. Key settings: base LR **0.003**, 8 epochs, batch 128, projection dimension 64; three folds with seed **20261025**; correction seeds **42/43/44**, 60 epochs, LR **0.01**, ranking weight **0.75**, KL weight **0.01**. Configuration and epoch/resource logs are in `code/train_config.json` and `code/output-training/`.

The recorded run used Python 3.12.14 / NumPy 2.5.3 / PyTorch 2.7.1+cu126, with a separate Python 3.11.3 / NumPy 1.24.3 / XGBoost 2.1.0 tree worker. A separate worker can be specified using `--tree-python PATH_TO_PYTHON_EXE`. Different library builds may produce small retraining differences.

## 3. Evaluation & Peer Review

**No retraining is required.** From `code/`, with the thread settings above, evaluate the supplied frozen checkpoint:

```cmd
python predict.py --method submission.py --checkpoint output --split validation --output runs/reproduced/validation-predictions.json
python score_predictions.py --split validation --predictions runs/reproduced/validation-predictions.json --output runs/reproduced/validation-score.json

python predict.py --method submission.py --checkpoint output --split test --output runs/reproduced/test-predictions.json
python score_predictions.py --split test --predictions runs/reproduced/test-predictions.json --output runs/reproduced/test-score.json
```

Expected test result: **394/467, accuracy 84.36830835117773%**. To evaluate a retrained checkpoint, replace `--checkpoint output` with `--checkpoint runs/retrained_final`.

- **Frozen checkpoint:** `code/output/model.npz` and `code/output/config.json`; also packaged in [model_bundle.zip](artifacts/model_bundle.zip).
- **Official evaluation JSON:** `code/output/validation-score.json` and `code/output/test-score.json`.
- **Report:** [English report](report/3036748919_MP2_He.docx).
- **Integrity and provenance:** `submission_manifest.json`, `records/final-lr003/`, and `code/output-training/`. Run `python verify_submission.py` from the repository root to verify matching files and checkpoint hashes.

Record the submitted code version with `git rev-parse HEAD`; use that commit in the code/checkpoint links. The model weight SHA-256 is:

```text
f9c730150364f9049a84cb21937d5f231dbb1755dc5eb27e8a624218a879241d
```

## 4. Reuse & AI Assistance

The official data, numerical parser and evaluation tools come from the DASE7506 MP2 starter package. GSM8K questions/references and course-provided Qwen3-0.6B candidate traces/embeddings retain their supplied licenses and notices in `code/data/`. The parser, scorer and supplied data are unchanged.

An AI coding assistant helped with implementation, debugging, error analysis, experiment organization and report drafting/translation. Training, evaluation and ablation records are retained for review.
