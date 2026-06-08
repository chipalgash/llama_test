# Repository Audit

Date: 2026-06-08

## Main Research Line

The defensible thesis experiment should use one base model for both languages:

`meta-llama/Meta-Llama-3-8B-Instruct`

The main comparison is:

| Language | Baseline | Fine-tuned |
| --- | --- | --- |
| English | Llama 3 8B Instruct | Llama 3 8B Instruct + EN QLoRA adapter |
| Russian | Llama 3 8B Instruct | Llama 3 8B Instruct + RU QLoRA adapter |

## Current Useful Inputs

| Area | File or directory | Status |
| --- | --- | --- |
| English raw stories | `data/creepypasta_stories_txt/` | Useful raw EN corpus |
| English prepared CSV | `data/processed/` | Useful current EN split/chunks |
| English Colab payload | `data/colab_payload/` and `data/horror_experiment_payload.zip` | Useful legacy payload |
| Russian raw stories | `data/russian_creepypasta_stories_md/` | Useful raw RU corpus |
| Russian prepared CSV | `data/processed_ru/` | Useful current RU split/chunks |
| Russian Colab payload | `data/russian_colab_payload/` and `data/russian_horror_experiment_payload.zip` | Useful legacy payload |
| English Llama notebook | `horror_llama3_payload_experiment_colab_FIXED_success.ipynb` | Best existing EN Llama 3 notebook |
| Next English notebook | `horror_llama3_payload_experiment_colab_NEXT_quality_run.ipynb` | Active draft notebook |
| Local evaluation script | `scripts/reevaluate_results.py` | Useful, but currently EN-oriented |

## Legacy or Non-main Files

| File or directory | Reason |
| --- | --- |
| `russian_horror_qwen_payload_experiment_colab.ipynb` | Qwen experiment; useful as a draft, but not part of the main thesis line |
| `experiment_1/` | Old detector-focused run |
| `experiment_2/` | Old detector-focused run |
| `results_success/` | Old successful EN run outputs; useful for reference only |
| `horror_llama3_payload_experiment_colab.ipynb` | Earlier notebook version |
| `horror_llama3_payload_experiment_colab_FIXED_success.ipynb` | Successful legacy notebook, superseded by the normalized structure |

## Immediate Gaps

1. The repository has CSV payloads, but not the required normalized JSONL files under `data/processed/en/` and `data/processed/ru/`.
2. The Russian main line still needs a Llama 3 notebook or script. The existing Russian notebook uses Qwen and must not be treated as the primary experiment.
3. Evaluation is partly detector-focused. The thesis line needs stylometry, statistical comparison, and manual evaluation templates as primary outputs.
4. README still describes a detection experiment rather than the final Llama 3 EN/RU QLoRA comparison.

## Current Structure Status

The repository now follows the thesis-facing layout:

```text
configs/
data/raw/en/
data/raw/ru/
data/processed/en/
data/processed/ru/
data/samples/
notebooks/
src/
outputs/en/
outputs/ru/
adapters/en_lora_adapter/
adapters/ru_lora_adapter/
archive/old_experiments/
```

Legacy notebooks, detector-focused runs, Qwen material, old reports, old scripts, zip payloads, and legacy CSV payloads were moved under `archive/old_experiments/` for traceability.

## Recommended Next Step

The CPU-side repository organization is complete. The next substantive step requires GPU access: run the EN/RU Colab notebooks to produce `baseline_generations.csv`, train the LoRA adapters, and generate `finetuned_generations.csv`.
