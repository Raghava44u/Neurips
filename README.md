# adversal2k

`adversal2k` is a numeric image-question benchmark for stable multimodal testing. The dataset rewrites open-ended scene questions into count-based questions with integer answers so the same image can be queried more consistently across repeated runs.

This repository contains the dataset, image files, evaluation code, model configuration files, and checkpoint layout used to test the benchmark.

## Main idea

The dataset is built for cases where free-form answers are too unstable. Instead of asking for a scene description, each sample asks for a numeric answer grounded in what is clearly visible in the image.

Typical examples look like this.

- How many umbrellas are in the foreground of the image
- Count only the clearly visible umbrellas in the image
- What integer equals the number of clearly visible umbrellas

This makes the benchmark easier to score and easier to reproduce.

## Main file

The main dataset file is:

- `adversal2k.json`

This is the file you should use for testing.

## Dataset size

- raw rows in `adversal2k.json`: `1990`
- effective evaluable rows in the compositional pipeline: `1924`

The smaller evaluated count happens because some rows do not contain the full compositional `port_new` structure expected by the loader.

## Data layout

Keep the files in this layout:

```text
MemEIC/
├── adversal2k.json
├── coco images/
├── CCKEB_images/
├── checkpoints/
├── easyeditor/
├── hparams/
├── hugging_cache/
├── requirements.txt
└── test_compositional_edit.py
```

The benchmark file uses relative image paths such as `coco images/000001.jpg`, so the image root should point to the repository folder itself.

## Important fields in the data

Each row contains the fields needed for visual edit, textual edit, locality checks, and portability checks.

Important fields include:

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

## How to use the data

### 1. Point evaluation to `adversal2k.json`

Use the dataset path through the environment variable:

```text
MEMEIC_EVAL_DATASET=...\MemEIC\adversal2k.json
```

### 2. Point the image root to the repository folder

Use:

```text
MEMEIC_IMAGE_ROOT=...\MemEIC
```

This is required because the JSON stores relative image paths.

### 3. Choose how many rows to test

Use:

```text
MEMEIC_TEST_NUM=1924
```

for the full evaluable benchmark.

If you want a quicker run, use a smaller value such as `100`, `250`, or `500`.

### 4. Choose whether to generate reports and artifacts

Use:

```text
MEMEIC_ENABLE_ARTIFACTS=0
```

for faster benchmark-only testing.

Use:

```text
MEMEIC_ENABLE_ARTIFACTS=1
```

if you also want predictions, reports, and visual outputs.

## Installation

### Requirements

- Python 3.11
- PyTorch 2.0.1 or compatible build
- CUDA GPU for practical evaluation

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Checkpoints

The test pipeline expects model assets and adapters to be available in the repository.

For the LLaVA run used here, the active stage 2 adapter directory is:

```text
checkpoints/llava_stage2
```

## How to test `adversal2k`

The main entrypoint is:

```text
test_compositional_edit.py
```

### Full evaluation command

Use this PowerShell command:

```powershell
Set-Location "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_DATASET = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\adversal2k.json"
$env:MEMEIC_IMAGE_ROOT = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_GPUS = "0"
$env:MEMEIC_EVAL_GAPS = "0"
$env:MEMEIC_RESULTS_DIR = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\results_numeric"
$env:MEMEIC_TEST_NUM = "1924"
$env:MEMEIC_ENABLE_ARTIFACTS = "0"
& "c:/Users/Dr-Prashantkumar/Downloads/Raghava-Personal/downlaod images/.venv/Scripts/python.exe" .\test_compositional_edit.py test_LLaVA_OURS_comp
```

### Faster small test

For a quick check on 100 rows:

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

### Medium-size test

For 250 rows:

```powershell
Set-Location "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_DATASET = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\adversal2k.json"
$env:MEMEIC_IMAGE_ROOT = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC"
$env:MEMEIC_EVAL_GPUS = "0"
$env:MEMEIC_EVAL_GAPS = "0"
$env:MEMEIC_RESULTS_DIR = "c:\Users\Dr-Prashantkumar\Downloads\Raghava-Personal\downlaod images\MemEIC\results_numeric_fast250"
$env:MEMEIC_TEST_NUM = "250"
$env:MEMEIC_ENABLE_ARTIFACTS = "0"
& "c:/Users/Dr-Prashantkumar/Downloads/Raghava-Personal/downlaod images/.venv/Scripts/python.exe" .\test_compositional_edit.py test_LLaVA_OURS_comp
```

## Output folders

Test outputs are written to the result directory you choose. Typical folders are:

```text
results_numeric/
├── failure_cases/
├── metrics/
├── models/
├── predictions/
├── reports/
└── visualizations/
```

If artifacts are disabled, the run skips the extra report pipeline and focuses on benchmark execution.

## Notes

- if `MEMEIC_EVAL_DATASET` is not set, the script falls back to `CCKEB_eval.json`
- if you want to test `adversal2k.json`, always set the dataset path explicitly
- GPU selection is controlled by `MEMEIC_EVAL_GPUS`
- on a single-GPU machine, active model execution runs on `cuda:0`
- full evaluation is slow because the benchmark performs sequential edit and sequential testing

## Summary

If your goal is to test a stable numeric visual benchmark, use `adversal2k.json`, point the image root to the repository, choose the number of rows you want to evaluate, and run `test_compositional_edit.py` with the environment variables shown above.