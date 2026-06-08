# Russian Horror Corpus Preparation Report

## Corpus

- Input Markdown files: 3833
- Stories after filtering/deduplication: 3798
- Skipped short stories: 28
- Skipped low-word/non-Russian stories: 1
- Skipped duplicate stories: 6
- Train stories: 3038
- Eval stories: 760
- Story chars: min=507, median=3929, mean=3408, p95=4977, max=5089
- Story words: min=79, median=616, mean=532, p95=793, max=872

## Categories

- Мистика и фантастика: 1821
- Истории из жизни: 1182
- Кошмары, вещие сны, сонный паралич: 795

## Chunks

- Full train chunks: 6830
- Full eval chunks: 1696
- Removed eval chunks due to exact train overlap: 0
- Payload train chunks: 2000
- Payload eval chunks: 500
- Chunk chars: min=700, median=1455, mean=1392, p95=1497, max=1500

## Leakage Checks

- Overlapping story hashes between train/eval: 0
- Overlapping chunk hashes between train/eval: 0

## Colab Payload

- Zip: `data/russian_horror_experiment_payload.zip`

Payload files:

- `train_with_descriptions.csv`
- `eval_with_descriptions.csv`
- `split_metadata.csv`
- `corpus_summary.csv`
- `russian_corpus_preparation_report.md`
