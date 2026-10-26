# DASE7506 MP2 — He Jiaxin

Reproducible offline candidate selection for DASE7506 MP2. Student ID: **3036748919**.

**Frozen final model:** base learning rate **0.003**, correction-head learning rate **0.01**. The supplied checkpoint reproduces predictions without training, a GPU, API calls, or additional pretrained weights.

| Split | Correct / total | Accuracy |
| --- | --- | --- |
| Validation, all questions | 748 / 798 | 93.7343% |
| Validation, disagreement questions | 305 / 355 | 85.9155% |
| Public test | 394 / 467 | **84.36830835117773%** |

These scores use `dase7506-mp2-open-hard-verifier-v5`, the official local evaluation protocol. The saved public scores document reproduction on the supplied bank; they are not presented as independent blind-test evidence.

## 1. Repository contents

| Path | Purpose |
| --- | --- |
| `code/submission.py` | Complete final NumPy inference implementation |
| `code/output/model.npz`, `config.json` | Frozen inference weights, transforms, statistics and configuration |
| `artifacts/model_bundle.zip` | Downloadable copy of exactly those two inference assets |
| `code/train.py`, `train_config.json` | Full training recipe from the supplied training split |
| `code/requirements.txt` | Pinned inference dependency |
| `code/requirements-training.txt` | Additional training dependencies |
| `code/predict.py`, `score_predictions.py`, `common.py` | Official runner, independent scorer and parser |
| `code/data/`, `guide/` | Supplied data, licenses and project guide |
| `code/output/*-score.json`, `*-predictions.json` | Original official evaluation outputs; not used by inference |
| `code/output-training/` | Training plan, epoch losses, fold indices, versions and resource summary |
| `records/final-lr003/` | Frozen checkpoint provenance and comparison records |
| `records/report-focused/` | Component-ablation records and initial learned baseline |
| `report/3036748919_MP2_He_English.docx` | **Eight-page English report** |
| `submission_manifest.json`, `verify_submission.py` | File hashes, asset accounting and local integrity check |

IDE files, Python caches, report working files and intermediate teacher weight caches are excluded from upload. The complete final trainer does not need those teacher caches: it fits them again from train.

## 2. Install for inference

Use **Python 3.12**. From a terminal:

```bat
git clone https://github.com/star4599/DASE7506-MP2-He_Jiaxin.git
cd DASE7506-MP2-He_Jiaxin
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r code/requirements.txt
python verify_submission.py
cd code
python check_data.py
```

The activation command above is for Windows CMD / Anaconda Prompt. In PowerShell, use `.\.venv\Scripts\Activate.ps1`; on Linux/macOS, use `source .venv/bin/activate`. If using an existing Python 3.12 environment such as `student`, skip creation and activation of `.venv`.

The official data is included in this repository. Inference needs only **NumPy 2.5.3**. After installation, prediction and scoring work offline. `verify_submission.py` checks file and checkpoint hashes; it does not train, predict, or inspect test questions.

## 3. Limit numerical libraries to four threads

Run the following before evaluation. All remaining prediction/training commands assume the current directory is **`code/`**.

Windows CMD / Anaconda Prompt:

```bat
set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
```

PowerShell:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
```

Linux/macOS:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

## 4. Reproduce the supplied checkpoint without training

Validation:

```bat
python predict.py --method submission.py --checkpoint output --split validation --output runs/reproduced/validation-predictions.json
python score_predictions.py --split validation --predictions runs/reproduced/validation-predictions.json --output runs/reproduced/validation-score.json
```

Public test:

```bat
python predict.py --method submission.py --checkpoint output --split test --output runs/reproduced/test-predictions.json
python score_predictions.py --split test --predictions runs/reproduced/test-predictions.json --output runs/reproduced/test-score.json
```

The scorer's `accuracy` is a percentage. Expected test result: **394 correct out of 467**, `accuracy: 84.36830835117773`. New outputs go into `runs/reproduced/`, preserving the original evaluation records in `output/`.

If downloading only the checkpoint, unzip `artifacts/model_bundle.zip` into a directory such as `model_bundle`, then replace `--checkpoint output` with `--checkpoint model_bundle`. The archive contains all required learned assets; no training or prediction cache is required.

For a supplied audit bank, use the same frozen method:

```bat
python predict.py --method submission.py --checkpoint output --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/audit-predictions.json
```

## 5. Train from scratch

Training additionally requires PyTorch 2.7.1 and XGBoost 2.1.0:

```bat
python -m pip install -r requirements-training.txt
python train.py --config train_config.json --output runs/retrained_final
```

The trainer reads train inputs and references only. It fits one full-training base model, three fold teachers, and three correction heads. It does not load the frozen checkpoint, reuse an existing teacher cache, or read validation/test during fitting.

A new output name is required if either the checkpoint directory or its accompanying training-log directory already exists. `runs/retrained_final/` contains the two resulting inference assets; `runs/retrained_final-training/` contains the training records. Validate a retrained model with:

```bat
python predict.py --method submission.py --checkpoint runs/retrained_final --split validation --output runs/retrained-validation-predictions.json
python score_predictions.py --split validation --predictions runs/retrained-validation-predictions.json --output runs/retrained-validation-score.json
```

An optional `--tree-python` accepts a separate Python interpreter with XGBoost. On the original machine, the command was equivalent to:

```bat
python train.py --config train_config.json --output runs/retrained_final_separate --tree-python D:\py\anaconda3\python.exe
```

The original main environment was Python 3.12.14 / NumPy 2.5.3 / PyTorch 2.7.1+cu126; the tree worker used Python 3.11.3 / NumPy 1.24.3 / XGBoost 2.1.0. Both ran on CPU. Different numerical-library builds may produce small retraining differences; the supplied frozen checkpoint is the reference for score reproduction without retraining.

Key settings are recorded in `train_config.json`: base seed 42, 8 linear epochs, learning rate 0.003, batch 128; 64-dimensional projection; 20 depth-4 tree rounds; three training folds with seed 20261025; correction seeds 42/43/44, 60 epochs, learning rate 0.01, weight decay 0.03, ranking weight 0.75 and KL weight 0.01.

## 6. Resource measurement

The two uncompressed inference assets total **291,329 bytes = 0.277832985 MiB**, below 16 MiB. Inference reads only these assets and the supplied inputs. Experiment checkpoints and saved predictions are not ensemble members or inference dependencies.

Recorded local hardware: **Intel Core i7-12700H, Windows 11, four CPU threads**. Full final-model retraining took **20.22 seconds** in the saved run. A recorded fresh validation prediction process plus independent scoring process took approximately **1.015 seconds**. Epoch losses, software versions and fold audits are in `code/output-training/`; ablation accuracy/resource records are in `records/report-focused/`.

The reference Xeon Platinum 8457C was not measured. The final checkpoint has no saved external full-test process timing; script-internal timings alone do not establish the official 30-second limit. To measure a new complete prediction/scoring pair, run this in PowerShell from `code/` after setting the four-thread variables:

```powershell
$elapsed = Measure-Command {
    python predict.py --method submission.py --checkpoint output --split test --output runs/timed/test-predictions.json
    if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
    python score_predictions.py --split test --predictions runs/timed/test-predictions.json --output runs/timed/test-score.json
    if ($LASTEXITCODE -ne 0) { throw 'Scoring failed' }
}
Get-Content runs/timed/test-score.json
$elapsed.TotalSeconds
```

Run processes sequentially and report the hardware alongside the measured wall time.

## 7. Matching code version and checkpoint

`submission_manifest.json` records the SHA-256 of the model, configuration, inference/training sources, English report and checkpoint archive. `.gitattributes` preserves exact file bytes across Git checkouts. Check integrity from the repository root:

```bat
python verify_submission.py
```

The weight SHA-256 is:

```text
f9c730150364f9049a84cb21937d5f231dbb1755dc5eb27e8a624218a879241d
```

Identify the submitted code version using the full commit that contains this manifest and its matching assets:

```bat
git rev-parse HEAD
```

Replace `<COMMIT_SHA>` below with that value when supplying permanent links:

- Code: `https://github.com/star4599/DASE7506-MP2-He_Jiaxin/tree/<COMMIT_SHA>`
- Checkpoint download: `https://github.com/star4599/DASE7506-MP2-He_Jiaxin/blob/<COMMIT_SHA>/artifacts/model_bundle.zip`

Uploading this repository does not create the separate course submission issue.

## 8. Reuse and AI assistance

The task, official data, numerical parser and evaluation tools originate from the DASE7506 MP2 starter package. Questions/references derive from GSM8K; supplied candidate traces and embeddings were prepared by the course using Qwen3-0.6B. Retain `code/data/LICENSE.txt` and `NOTICE.txt`. No external training dataset or additional pretrained weights are used by this method.

An AI coding assistant helped with implementation, debugging, experiment organization, error discussion and report drafting/translation. Training, evaluation and ablation records are retained for review. The English report discusses unsuccessful choices, regressions and accuracy/resource trade-offs; the saved provenance records document the development history.
