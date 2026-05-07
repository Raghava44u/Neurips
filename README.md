# MemEIC

MemEIC is a vision-language knowledge editing framework for continual and compositional editing. This repository contains the evaluation code, model configuration files, checkpoints layout, and benchmark assets used to run MemEIC on both the original CCKEB benchmark and a numeric adversarial benchmark built for stable image-grounded answers.

This version of the repository is centered on dataset-driven evaluation. The main addition is a numeric adversarial benchmark in which each example is rewritten as a counting question with an integer ground truth, so repeated prompts against the same image are less likely to drift semantically.

## What This Repository Contains

- MemEIC evaluation code in `easyeditor/`
- MemEIC evaluation entrypoint in `test_compositional_edit.py`
- hyperparameter files in `hparams/`
- MemEIC checkpoints under `checkpoints/`
- the numeric adversarial benchmark in `adversal2k.json`
- image assets in `coco images/` and `CCKEB_images/`

## Dataset

## Main Dataset in This Repository

The most important dataset in this repository is:

- `adversal2k.json`

This file is a numeric adversarial benchmark designed for stable multimodal evaluation. Each row is structured so the answer is numeric, typically a count, instead of an open-ended scene description.

The goal is simple:

- reduce answer variability across repeated runs
- make ground truth easier to verify
- test whether MemEIC can preserve edited knowledge under compositional prompting

### Numeric Adversarial Benchmark

The benchmark uses image-grounded prompts such as:

- how many umbrellas are in the foreground of the image
- count only the clearly visible umbrellas in the image
- what integer equals the number of clearly visible umbrellas

Each example contains fields used by MemEIC for visual edit, textual edit, locality checks, and compositional portability. Important fields include:

- `src`
- `rephrase`
- `pred`
- `alt`
- `image`
- `loc`
- `loc_ans`
- `m_loc`
- `m_loc_q`
- `m_loc_a`
- `port_new`
- `textual_edit`
- `original_question`
- `rewritten_question`
- `original_ground_truth`
- `improved_ground_truth`
- `reasoning_type`
- `why_this_question_is_more_stable`
- `count_object`

### Dataset Size

- raw rows in `adversal2k.json`: `1990`
- effective evaluable samples loaded by the compositional MemEIC pipeline: `1924`

The difference exists because a subset of rows does not contain the compositional `port_new` structure required by the loader.

### Image Layout

The numeric adversarial benchmark expects images under:

```text
MemEIC/
├── adversal2k.json
├── coco images/
└── CCKEB_images/
```

For `adversal2k.json`, the active image root is usually the repository root itself:

```text
MEMEIC_IMAGE_ROOT=...\MemEIC
```

because the JSON contains relative paths such as `coco images/000001.jpg`.

## Original CCKEB Benchmark

The repository still supports the original CCKEB files:

- `CCKEB_eval.json`
- `CCKEB_images/mmkb_images/`

If no environment override is provided, `test_compositional_edit.py` defaults to `CCKEB_eval.json`. If you want to evaluate the numeric adversarial benchmark, you must set `MEMEIC_EVAL_DATASET` explicitly.

## Installation

### Requirements

- Python 3.11
- PyTorch 2.0.1 or compatible build
- CUDA-enabled GPU for practical evaluation

### Install

```bash
git clone https://github.com/MemEIC/MemEIC.git
cd MemEIC

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Checkpoints and Model Assets

Place the required model assets in the expected folders before evaluation.

Typical directories used by this repository include:

```text
MemEIC/
├── checkpoints/
├── hugging_cache/
└── hparams/
```

For the MemEIC LLaVA evaluation used in this repository, the active stage 2 adapter path is:

```text
checkpoints/llava_stage2
```

## Evaluation

## Main Evaluation Script

The main entrypoint is:

```text
test_compositional_edit.py
```

It supports both default CCKEB evaluation and environment-controlled custom evaluation.

### Evaluate the Numeric Adversarial Benchmark

PowerShell command:

```powershell
Set-Location "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_DATASET = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\adversal2k.json"
$env:MEMEIC_IMAGE_ROOT = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_GPUS = "0"
$env:MEMEIC_EVAL_GAPS = "0"
$env:MEMEIC_RESULTS_DIR = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\results_numeric"
$env:MEMEIC_TEST_NUM = "1924"
& "c:/Users/Dr-Prashantkumar/Downloads/Raghava-Personal/downlaod images/.venv/Scripts/python.exe" .\test_compositional_edit.py test_LLaVA_OURS_comp
```

### Faster Evaluation Without Artifact Generation

Artifact generation adds extra work for predictions, reports, visualizations, and failure-case summaries. To run a faster benchmark-only evaluation, disable artifacts:

```powershell
Set-Location "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_DATASET = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\adversal2k.json"
$env:MEMEIC_IMAGE_ROOT = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_GPUS = "0"
$env:MEMEIC_EVAL_GAPS = "0"
$env:MEMEIC_RESULTS_DIR = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\results_numeric_noartifacts"
$env:MEMEIC_TEST_NUM = "1924"
$env:MEMEIC_ENABLE_ARTIFACTS = "0"
& "c:/Users/Dr-Prashantkumar/Downloads/Raghava-Personal/downlaod images/.venv/Scripts/python.exe" .\test_compositional_edit.py test_LLaVA_OURS_comp
```

### Small Subset Evaluation

If you want a quicker sanity check, reduce `MEMEIC_TEST_NUM`.

Example for 100 samples:

```powershell
Set-Location "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_DATASET = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\adversal2k.json"
$env:MEMEIC_IMAGE_ROOT = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_GPUS = "0"
$env:MEMEIC_EVAL_GAPS = "0"
$env:MEMEIC_RESULTS_DIR = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\results_numeric_fast100"
$env:MEMEIC_TEST_NUM = "100"
$env:MEMEIC_ENABLE_ARTIFACTS = "0"
& "c:/Users/Dr-Prashantkumar/Downloads/Raghava-Personal/downlaod images/.venv/Scripts/python.exe" .\test_compositional_edit.py test_LLaVA_OURS_comp
```

## What the Evaluation Produces

Depending on settings, MemEIC writes outputs under the results directory you choose. Typical folders include:

```text
results_numeric/
├── failure_cases/
├── metrics/
├── models/
├── predictions/
├── reports/
└── visualizations/
```

If `MEMEIC_ENABLE_ARTIFACTS=0`, the run skips the extra artifact pipeline and focuses on benchmark execution and core result files.

## Important Notes for This Repository

- `test_compositional_edit.py` defaults to `CCKEB_eval.json` only when `MEMEIC_EVAL_DATASET` is not set
- GPU selection is controlled by `MEMEIC_EVAL_GPUS`
- on a single-GPU machine, MemEIC runs the active model on `cuda:0` and uses CPU for cache-side storage
- the numeric benchmark is much slower than a standard feed-forward test because it performs sequential edit and evaluation across the benchmark

## Repository Structure

```text
MemEIC/
├── README.md
├── adversal2k.json
├── CCKEB_images/
├── checkpoints/
├── coco images/
├── easyeditor/
├── figs/
├── hparams/
├── hugging_cache/
├── requirements.txt
└── test_compositional_edit.py
```

## Citation

If you use MemEIC, CCKEB, or the numeric adversarial benchmark adaptation in this repository, cite the original MemEIC paper and clearly describe any benchmark modifications you made for numeric evaluation.