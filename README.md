# Text2Cypher Experiments (Thesis Repo)

This repo accompanies my thesis:

> **“Think Beyond LLM: Can Agentic Workflow outperform Fine Tuned LLM in terms of Text2Cypher?”** (GraphML Lab)

If you want the *full* experimental context (datasets, Neo4j instances/data import, evaluation details), the thesis is the source of truth.

This README is a **suggested cookbook**: which folders to use, what to install, and which scripts to run.

---

## Repo layout (what you will actually run)

- `fine-tuning/`
  - LoRA fine-tuning scripts (+ DeepSpeed config in-code)
  - Includes `merge_peft.py` helper
- `vllm_test_0.9/`
  - Minimal vLLM “OpenAI-compatible server” setup (the name is historical)
- `test/`
  - Benchmark scripts for:
    - original/base LLM
    - fine-tuned LLM
    - agentic workflow
- `benchmark_dataset/benchmark_dataset.json`
  - The benchmark dataset used by the scripts in `test/`
- `text2cypher-cleanup-public/`
  - Separate package used in this project (not required for the minimal benchmark runs unless you are extending the work)

---

## Before you start (important constraints)

### OS / Hardware assumptions

- Most scripts and dependency pins are written with **Linux + NVIDIA GPU (CUDA)** in mind.
- Fine-tuning requires a GPU (and *a lot* of VRAM depending on model size).
- **Docker** is required for the Neo4j multi-instance setup.

### Tooling

- Python **3.11** (the `fine-tuning/pyproject.toml` is `>=3.11,<3.12`)
- [`uv`](https://github.com/astral-sh/uv) for environment management
- Docker for Neo4j

### Absolute paths are hard-coded in scripts

Many scripts contain absolute paths like:

- `root_dir_you_like/projects/llm/...`
- `root_dir_you_like/projects/test/...`
- `root_dir_you_like/projects/fine-tuning/...`

To reproduce experiments, you must do **one** of the following:

1. **Edit the scripts** to point to your local paths (recommended), **or**
2. **Mirror the directory layout** on your machine (works best on Linux).

Example (Linux) to mirror the layout:

```bash
sudo mkdir -p root_dir_you_like/projects/{llm,test,fine-tuning}
sudo chown -R "$USER":"$USER" root_dir_you_like/projects

# Put the benchmark dataset where the test scripts expect it
cp benchmark_dataset/benchmark_dataset.json root_dir_you_like/projects/test/benchmark_dataset.json
```

---

## Step 0 — Get the benchmark dataset in place

The benchmark dataset is in:

```text
benchmark_dataset/benchmark_dataset.json
```

The scripts in `test/` currently read it from:

```text
root_dir_you_like/projects/test/benchmark_dataset.json
```

So either copy it there (see snippet above) or change the path inside the benchmark scripts.

---

## Step 1 — Start Neo4j instances (required for evaluation)

Both fine-tuning (MTQ variant) and benchmarking validate queries by executing them against Neo4j.

### Credentials/ports are hard-coded

The scripts use:

- username: `neo4j`
- password: `password`
- multiple local bolt ports (see mappings below)

If you do not want to use that password, you must edit the scripts accordingly.

### Benchmark (5 DBs)

The benchmark scripts use this mapping:

```python
{
  "police_investigation_crime": "7683",
  "network_management": "7684",
  "movie_director_show_actor": "7685",
  "drugs_proteins_diseases": "7686",
  "food_ingredients_allergens": "7687",
}
```

You need **5 Neo4j instances**, each reachable at `neo4j://localhost:<port>`.

### MTQ fine-tuning (5 DBs)

`fine-tuning/lora_fine_tuning_with_ds_mtq.py` uses another mapping:

```python
{"wwc": 7678, "er": 7680, "covid": 7681, "bloom50": 7682, "healthcare": 7679}
```

So MTQ fine-tuning also expects **5 Neo4j instances** on ports **7678–7682**.

### Data import

You must import the correct data into each DB instance.
The data importers and exact datasets are described in the thesis.
(Yes, this part is tedious.)

---

## Step 2 — (Optional but typical) Run vLLM server for local inference

Benchmarking uses the **OpenAI Python SDK** pointed at a local server:

- base URL: `http://localhost:8000/v1`
- API key: `abc` (dummy)

`vllm_test_0.9/vllm_serve.sh` currently contains hard-coded values:

```bash
export CUDA_VISIBLE_DEVICES=1
vllm serve root_dir_you_like/projects/llm/deepseek-coder-33b-instruct \
  --dtype auto --api-key abc --max-model-len 2048 \
  --gpu-memory-utilization 0.96 --trust-remote-code
```

To use it:

```bash
cd vllm_test_0.9
uv sync

# Edit vllm_serve.sh to point to your model path and GPU index
bash vllm_serve.sh
```

Notes:

- vLLM installation can be the hardest part; many issues are dependency/toolchain related.
- If vLLM hangs/crashes mid-run, rerunning often works.

---

## Step 3 — Fine-tune (LoRA)

All fine-tuning code lives in `fine-tuning/`.

### Install deps

```bash
cd fine-tuning
uv sync
```

`fine-tuning/pyproject.toml` pins Torch to a specific CUDA build on Linux.
You may need to adjust Torch/CUDA versions to match your GPU and driver.

### Fine-tuning script (non-MTQ)

Script: `fine-tuning/lora_fine_tuning_with_ds.py`

What you must change before running:

- `path_to_a_folder/...` (model path, dataset paths, output paths)
- the `start_train("your_model_name_goes_here", ...)` call

Run:

```bash
uv run python lora_fine_tuning_with_ds.py
```

### Fine-tuning script (MTQ)

Script: `fine-tuning/lora_fine_tuning_with_ds_mtq.py`

This script:

- reads MTQ train/eval from JSON (`train_mtq.json`, `eval_mtq.json`)
- validates generated Cypher by executing against Neo4j during eval

What you must change before running:

- absolute paths to datasets and model directory
- Neo4j auth/ports if your setup differs

Run:

```bash
uv run python lora_fine_tuning_with_ds_mtq.py
```

### Merging LoRA adapter

At the end of training, the scripts attempt to merge the LoRA adapter into the base model via `merge_and_unload()`.
Sometimes this fails.

If it fails, use:

- `fine-tuning/merge_peft.py` (edit paths inside) to merge explicitly.

### Common fine-tuning failures

- Errors like **"current loss scale already in minimum"**: re-run; training is stochastic.
- Mid-run crashes: re-run.

---

## Step 4 — Run benchmarks (original LLM vs fine-tuned vs agentic workflow)

All benchmark scripts live in `test/`.

### Install deps

In this folder I used `requirements.txt` for reproducibility.
If `uv` complains about dependency inconsistency, prefer the pinned `requirements.txt`.

```bash
cd test

# one way (pip-compat):
uv venv
uv pip install -r requirements.txt
```

### Make sure prerequisites are running

Before running any benchmark script, you need:

1. Neo4j instances up and loaded with the benchmark data (ports 7683–7687)
2. vLLM running at `http://localhost:8000/v1` (or change `base_url` in scripts)
3. `benchmark_dataset.json` accessible at the path used in the scripts

### Benchmark: base/original LLM

Script: `test/benchmark_original_llm.py`

What you must edit:

- `model_name` (and possibly the model path prefix)
- output path (`results_of_experiments.txt`) if you don’t have `root_dir_you_like/projects/test/`

Run:

```bash
uv run python benchmark_original_llm.py
```

### Benchmark: fine-tuned LLM

Script: `test/benchmark_fine_tuned_llm.py`

What you must edit:

- `model_name` to match your merged fine-tuned model directory
  (the script expects it under `root_dir_you_like/projects/lora-fine-tuned-llm-mtq/`)

Run:

```bash
uv run python benchmark_fine_tuned_llm.py
```

### Benchmark: agentic workflow

Script: `test/agentic_workflow_and_benchmark.py`

What you must edit:

- `model_name` at the bottom of the file
- any absolute file paths

Run:

```bash
uv run python agentic_workflow_and_benchmark.py
```

---

## Why are some folder names confusing?

Because folder names are referenced by lock/config files (`uv.lock`, `pyproject.toml`) and renaming can break reproduction.

---

## If you are stuck

If you are trying to reproduce this work, the “sharp edges” are usually:

1. vLLM installation/toolchain
2. Matching Torch/CUDA versions
3. Setting up multiple Neo4j instances + data import
4. Hard-coded absolute paths and credentials inside scripts
