# Llama 3 Horror/Thriller QLoRA Experiment

This project is a research prototype for a master's thesis. The goal is not to build a production-ready horror text generator, but to compare baseline and QLoRA-adapted Llama 3 8B Instruct on genre-specific generation in English and Russian.

## Research Goal

The practical experiment compares how well one base model generates horror/thriller prose before and after small QLoRA adaptation on genre corpora.

Main model:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

## Experimental Design

| Language | Baseline | Fine-tuned |
| --- | --- | --- |
| English | Llama 3 8B Instruct | Llama 3 8B Instruct + EN LoRA adapter |
| Russian | Llama 3 8B Instruct | Llama 3 8B Instruct + RU LoRA adapter |

Qwen, detector-only runs, and older notebooks are kept only as legacy material. They are not part of the main thesis line.

## Repository Layout

```text
configs/                 # EN/RU experiment and generation settings
data/processed/en/       # normalized EN train/eval JSONL payloads
data/processed/ru/       # normalized RU train/eval JSONL payloads
notebooks/               # thesis-facing Colab notebooks
src/data/                # payload conversion scripts
src/generation/          # fixed prompts and generation helpers
src/evaluation/          # stylometry, statistics, manual eval templates
outputs/                 # generation and evaluation CSVs
adapters/                # EN/RU LoRA adapter directories
archive/old_experiments/ # legacy notebooks and old experiment material
reports/audit/           # repository audit notes
```

## Data Preparation

The current small reproducible payload uses:

```text
500 train chunks per language
100 eval prompts per language
story-level train/eval split
```

Regenerate normalized JSONL payloads from the existing CSV artifacts:

```bash
python3 src/data/convert_payload_csv_to_jsonl.py --language all --max-train 500 --max-eval 100
```

Expected outputs:

```text
data/processed/en/train_payload.jsonl
data/processed/en/eval_payload.jsonl
data/processed/ru/train_payload.jsonl
data/processed/ru/eval_payload.jsonl
```

## Baseline Generation

Use the same prompts and generation settings for English and Russian baseline generation. Settings are stored in:

```text
configs/generation_config.yaml
```

Required output files:

```text
outputs/en/baseline_generations.csv
outputs/ru/baseline_generations.csv
```

Colab/GPU commands:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_baseline \
  --language en \
  --eval-jsonl data/processed/en/eval_payload.jsonl \
  --output-csv outputs/en/baseline_generations.csv \
  --max-samples 100

PYTHONPATH=$PWD python -m src.generation.generate_baseline \
  --language ru \
  --eval-jsonl data/processed/ru/eval_payload.jsonl \
  --output-csv outputs/ru/baseline_generations.csv \
  --max-samples 100
```

## QLoRA Fine-Tuning

Initial QLoRA settings are intentionally small for Colab:

```text
LoRA r = 8
LoRA alpha = 16
LoRA dropout = 0.05
epochs = 1
learning rate = 1e-5
batch size = 1
gradient accumulation = 8
max sequence length = 1024
target modules = q_proj, v_proj
```

Fine-tuned adapters should be saved to:

```text
adapters/en_lora_adapter/
adapters/ru_lora_adapter/
```

Colab/GPU training commands:

```bash
PYTHONPATH=$PWD python -m src.training.train_qlora \
  --train-jsonl data/processed/en/train_payload.jsonl \
  --eval-jsonl data/processed/en/eval_payload.jsonl \
  --adapter-dir adapters/en_lora_adapter \
  --max-train-samples 500 \
  --max-eval-samples 100

PYTHONPATH=$PWD python -m src.training.train_qlora \
  --train-jsonl data/processed/ru/train_payload.jsonl \
  --eval-jsonl data/processed/ru/eval_payload.jsonl \
  --adapter-dir adapters/ru_lora_adapter \
  --max-train-samples 500 \
  --max-eval-samples 100
```

Fine-tuned generations should be saved to:

```text
outputs/en/finetuned_generations.csv
outputs/ru/finetuned_generations.csv
```

Colab/GPU fine-tuned generation commands:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_with_adapter \
  --language en \
  --eval-jsonl data/processed/en/eval_payload.jsonl \
  --adapter-dir adapters/en_lora_adapter \
  --output-csv outputs/en/finetuned_generations.csv \
  --max-samples 100

PYTHONPATH=$PWD python -m src.generation.generate_with_adapter \
  --language ru \
  --eval-jsonl data/processed/ru/eval_payload.jsonl \
  --adapter-dir adapters/ru_lora_adapter \
  --output-csv outputs/ru/finetuned_generations.csv \
  --max-samples 100
```

## Evaluation

Primary evaluation layers:

1. Stylometric features.
2. Mann-Whitney U statistical comparisons and rank-biserial effect size.
3. Manual expert evaluation templates.
4. Qualitative analysis of generated examples.

Run stylometry after baseline and fine-tuned CSV files exist:

```bash
python3 -m src.evaluation.run_stylometry \
  --language en \
  --human-jsonl data/processed/en/eval_payload.jsonl \
  --baseline-csv outputs/en/baseline_generations.csv \
  --finetuned-csv outputs/en/finetuned_generations.csv \
  --features-csv outputs/en/stylometric_features.csv \
  --summary-csv outputs/en/metric_summary.csv

python3 -m src.evaluation.run_stylometry \
  --language ru \
  --human-jsonl data/processed/ru/eval_payload.jsonl \
  --baseline-csv outputs/ru/baseline_generations.csv \
  --finetuned-csv outputs/ru/finetuned_generations.csv \
  --features-csv outputs/ru/stylometric_features.csv \
  --summary-csv outputs/ru/metric_summary.csv
```

Create manual evaluation templates:

```bash
python3 -m src.evaluation.manual_eval_template
```

## Known Limitations

Llama 3 8B Instruct is stronger in English than Russian. Russian generation may be less stable. Small-corpus QLoRA may not improve every metric. Automatic stylometry does not fully measure literary quality, so manual expert evaluation is required. The experiment is constrained by Colab/GPU resources.
