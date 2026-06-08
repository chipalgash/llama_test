# Инструкция для Codex: приведение NLP-эксперимента по Llama 3 8B к защищаемому формату магистерской диссертации

## 1. Роль агента

Ты работаешь как senior NLP engineer и research engineering assistant.
Твоя задача — помочь привести исследовательский репозиторий к аккуратному, воспроизводимому и защищаемому виду для магистерской диссертации на тему:

> «Настройка языковой модели для генерации текстов в жанрах “ужасы” и “триллер”».

Проект не должен превращаться в production-систему. Главная цель — провести контролируемый научный эксперимент:

> Сравнить качество жанровой генерации модели Llama 3 8B Instruct в двух языках — английском и русском — до и после QLoRA-адаптации на жанровых корпусах horror/thriller.

## 2. Основная исследовательская логика

Нужно реализовать и организовать экспериментальную схему:

| Язык       | Baseline                           | Fine-tuned                            |
| ---------- | ---------------------------------- | ------------------------------------- |
| Английский | Llama 3 8B Instruct без дообучения | Llama 3 8B Instruct + EN LoRA adapter |
| Русский    | Llama 3 8B Instruct без дообучения | Llama 3 8B Instruct + RU LoRA adapter |

Важно:

* Базовая модель должна быть одна и та же: `meta-llama/Meta-Llama-3-8B-Instruct`.
* Не использовать Qwen, Mistral, Saiga или другие модели в основной экспериментальной линии.
* Если в репозитории есть эксперименты с Qwen, не удалять их без необходимости, но вынести в архивную/черновую зону или явно пометить как неосновные.
* Цель не в том, чтобы получить идеальную horror-модель, а в том, чтобы получить измеримое сравнение baseline и QLoRA-adapted модели.

## 3. Текущая проблема проекта

Сейчас проект частично расползся:

* есть англоязычный pipeline под Llama 3 8B;
* есть русскоязычные наработки, но часть из них может быть ориентирована на Qwen;
* дообучение получается долгим и нестабильным;
* нужно привести код к простому, понятному и воспроизводимому виду;
* нужно получить CSV/таблицы/метрики, которые потом можно использовать в тексте диссертации.

Твоя задача — не усложнять проект, а упростить его.

## 4. Главные принципы работы

При внесении изменений придерживайся следующих принципов:

1. Минимальный жизнеспособный эксперимент важнее идеального обучения.
2. Все результаты должны сохраняться в понятные CSV/JSON-файлы.
3. Все параметры эксперимента должны быть вынесены в конфиги.
4. Код должен быть понятен человеку, который будет описывать его в диссертации.
5. Не добавлять сложные зависимости без необходимости.
6. Не переписывать весь проект с нуля, если можно аккуратно реорганизовать существующую логику.
7. Не менять исследовательскую постановку без явного основания.
8. Не запускать обучение на огромных датасетах по умолчанию.
9. Сначала должна работать маленькая версия эксперимента.
10. Все notebook/script outputs должны быть воспроизводимыми.

## 5. Рекомендуемая структура репозитория

Приведи репозиторий примерно к такой структуре:

```text
llama_test/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── en_experiment.yaml
│   ├── ru_experiment.yaml
│   └── generation_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── en/
│   │   └── ru/
│   │
│   ├── processed/
│   │   ├── en/
│   │   │   ├── train_payload.jsonl
│   │   │   ├── eval_payload.jsonl
│   │   │   └── metadata.json
│   │   │
│   │   └── ru/
│   │       ├── train_payload.jsonl
│   │       ├── eval_payload.jsonl
│   │       └── metadata.json
│   │
│   └── samples/
│       ├── en_sample.jsonl
│       └── ru_sample.jsonl
│
├── notebooks/
│   ├── 01_en_llama3_baseline_and_qlora.ipynb
│   ├── 02_ru_llama3_baseline_and_qlora.ipynb
│   └── 03_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── prepare_en_payload.py
│   │   ├── prepare_ru_payload.py
│   │   └── split_utils.py
│   │
│   ├── generation/
│   │   ├── generate_baseline.py
│   │   ├── generate_with_adapter.py
│   │   └── prompts.py
│   │
│   ├── training/
│   │   ├── train_qlora.py
│   │   └── lora_config.py
│   │
│   ├── evaluation/
│   │   ├── stylometry_en.py
│   │   ├── stylometry_ru.py
│   │   ├── metrics.py
│   │   ├── statistical_tests.py
│   │   └── manual_eval_template.py
│   │
│   └── utils/
│       ├── io.py
│       ├── seeds.py
│       └── logging.py
│
├── outputs/
│   ├── en/
│   │   ├── baseline_generations.csv
│   │   ├── finetuned_generations.csv
│   │   ├── stylometric_features.csv
│   │   ├── metric_summary.csv
│   │   └── manual_eval_sample.csv
│   │
│   └── ru/
│       ├── baseline_generations.csv
│       ├── finetuned_generations.csv
│       ├── stylometric_features.csv
│       ├── metric_summary.csv
│       └── manual_eval_sample.csv
│
├── adapters/
│   ├── en_lora_adapter/
│   └── ru_lora_adapter/
│
└── archive/
    └── old_experiments/
```

Если полная реструктуризация слишком рискованна, сначала создай эту структуру параллельно и перенеси туда только актуальные файлы.

## 6. Основная модель

Во всех основных экспериментах использовать:

```python
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
```

Эта модель должна использоваться:

* для английского baseline;
* для русского baseline;
* для английского QLoRA fine-tuning;
* для русского QLoRA fine-tuning;
* для генерации после подключения LoRA adapter.

Не использовать разные base models для разных языков.

## 7. Рекомендуемые параметры QLoRA

Сделать маленький, воспроизводимый эксперимент.
Начальные параметры:

```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1024  # или 1536, если хватает памяти

TARGET_MODULES = ["q_proj", "v_proj"]
```

Если обучение слишком медленное или нестабильное:

* уменьшить `MAX_SEQ_LENGTH` до `1024`;
* уменьшить число train chunks;
* оставить только `q_proj`, `v_proj`;
* не увеличивать epochs;
* не пытаться обучать на всём датасете.

Первый рабочий вариант должен использовать:

```text
300–500 train chunks на язык
50–100 eval prompts на язык
1 epoch
```

Только после получения стабильных результатов можно увеличить до:

```text
800–1000 train chunks на язык
```

## 8. Подготовка данных

Нужно обеспечить разбиение на уровне stories, а не случайных chunks.

Правильно:

```text
story_1 -> train
story_2 -> train
story_3 -> eval
```

Неправильно:

```text
chunk_1 from story_1 -> train
chunk_2 from story_1 -> eval
```

Нужно избежать leakage между train и eval.

Для каждого языка подготовить:

```text
data/processed/en/train_payload.jsonl
data/processed/en/eval_payload.jsonl

data/processed/ru/train_payload.jsonl
data/processed/ru/eval_payload.jsonl
```

Каждая строка JSONL должна иметь примерно такую структуру:

```json
{
  "id": "en_000001",
  "story_id": "story_001",
  "language": "en",
  "genre": "horror",
  "prompt": "Write a short horror/thriller scene...",
  "text": "The generated or target continuation...",
  "source": "dataset_name"
}
```

Для русского:

```json
{
  "id": "ru_000001",
  "story_id": "story_001",
  "language": "ru",
  "genre": "horror",
  "prompt": "Напиши короткую сцену в жанре ужасов/триллера...",
  "text": "Текст обучающего фрагмента...",
  "source": "dataset_name"
}
```

## 9. Промпты для генерации

Для baseline и fine-tuned генерации использовать одинаковые prompts.

### Английский prompt

```text
Write a short literary scene in the horror/thriller genre.
Focus on atmosphere, suspense, sensory details, and psychological tension.
Do not explain the horror directly. Show a coherent scene.
```

### Русский prompt

```text
Напиши короткую литературную сцену в жанре ужасов/триллера.
Сосредоточься на атмосфере, напряжении, сенсорных деталях и психологической тревоге.
Не объясняй ужас напрямую. Покажи связную сцену.
```

Важно:

* не менять prompts между baseline и fine-tuned;
* сохранять prompt вместе с генерацией;
* фиксировать параметры генерации.

## 10. Параметры генерации

Использовать одинаковые параметры для всех четырёх условий эксперимента:

```python
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DO_SAMPLE = True
```

Для воспроизводимости:

```python
SEED = 42
```

Если генерации слишком длинные или модель уходит в повторы, можно снизить:

```python
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
```

Но параметры должны быть одинаковыми для baseline и fine-tuned в рамках одного сравнения.

## 11. Что нужно сохранить после генерации

Для каждого языка и каждого состояния модели сохранить:

```text
outputs/en/baseline_generations.csv
outputs/en/finetuned_generations.csv

outputs/ru/baseline_generations.csv
outputs/ru/finetuned_generations.csv
```

Минимальные колонки:

```text
id
language
condition
prompt
generated_text
model_name
adapter_name
generation_temperature
top_p
max_new_tokens
seed
```

Где `condition`:

```text
human
baseline
finetuned
```

## 12. Оценка качества

Реализовать три уровня оценки.

### 12.1. Автоматическая стилометрия

Для английского и русского отдельно посчитать:

```text
char_count
word_count
sentence_count
avg_word_length
type_token_ratio
question_mark_count
exclamation_mark_count
ellipsis_count
fear_word_rate
suspense_word_rate
cliche_count
repetition_score
```

Для английского использовать английскую токенизацию words.

Для русского обязательно использовать кириллицу:

```python
import re

words = re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", text.lower())
```

Не использовать английский regex для русских текстов.

### 12.2. Статистическое сравнение

Сравнить группы:

```text
human vs baseline
human vs finetuned
baseline vs finetuned
```

Для каждого языка отдельно.

Методы:

```text
Mann–Whitney U test
rank-biserial correlation или другой effect size
mean
median
std
```

Результаты сохранить в:

```text
outputs/en/metric_summary.csv
outputs/ru/metric_summary.csv
```

### 12.3. Шаблон для ручной экспертной оценки

Создать CSV для ручной оценки:

```text
outputs/en/manual_eval_sample.csv
outputs/ru/manual_eval_sample.csv
```

Колонки:

```text
id
language
condition
prompt
generated_text
genre_score_1_5
suspense_score_1_5
language_naturalness_1_5
coherence_1_5
originality_1_5
comment
```

Выбрать случайно:

```text
10 EN baseline
10 EN finetuned
10 RU baseline
10 RU finetuned
```

Желательно сделать blinded version, где condition скрыт:

```text
outputs/manual_eval_blinded.csv
```

## 13. Не использовать AI detector как главный результат

Если в старом коде есть supervised detector или классификатор human/base/fine-tuned, его можно оставить как вспомогательный эксперимент, но не делать главным доказательством качества.

Главные результаты:

1. стилометрические признаки;
2. статистические различия;
3. ручная экспертная оценка;
4. качественный анализ примеров генерации.

## 14. README

Обновить README так, чтобы он объяснял проект как научный эксперимент.

README должен включать:

```text
1. Research goal
2. Experimental design
3. Model
4. Datasets
5. Data preparation
6. Baseline generation
7. QLoRA fine-tuning
8. Fine-tuned generation
9. Evaluation
10. Outputs
11. How to reproduce
12. Known limitations
```

Особенно важно явно написать:

```text
This project is a research prototype for a master's thesis. 
The goal is not to build a production-ready horror text generator, but to compare baseline and QLoRA-adapted Llama 3 8B Instruct on genre-specific generation in English and Russian.
```

## 15. Ограничения, которые нужно учитывать

В коде и README желательно отразить:

* Llama 3 8B Instruct изначально сильнее ориентирована на английский язык;
* русская генерация может быть менее стабильной;
* QLoRA-адаптация на малом корпусе может не улучшить все характеристики;
* автоматические метрики не полностью отражают литературное качество;
* ручная оценка нужна как дополнительный уровень анализа;
* эксперимент ограничен ресурсами Colab/GPU.

## 16. Конкретный порядок задач

Выполняй задачи в таком порядке.

### Этап 1. Аудит репозитория

1. Найди все notebook/scripts, связанные с:

   * английским payload;
   * русским payload;
   * Llama 3;
   * Qwen;
   * QLoRA;
   * generation;
   * evaluation.
2. Составь краткий список актуальных и устаревших файлов.
3. Ничего не удаляй без необходимости.
4. Устаревшие файлы можно перенести в `archive/old_experiments/`.

### Этап 2. Организация структуры

1. Создай папки:

   * `configs/`
   * `src/`
   * `notebooks/`
   * `outputs/`
   * `adapters/`
   * `archive/`
2. Перенеси или скопируй актуальные scripts/notebooks в новую структуру.
3. Обнови пути в коде.

### Этап 3. Английский baseline

1. Подготовь `eval_payload.jsonl`.
2. Сгенерируй 50–100 baseline outputs через `meta-llama/Meta-Llama-3-8B-Instruct`.
3. Сохрани в `outputs/en/baseline_generations.csv`.

### Этап 4. Русский baseline

1. Подготовь `eval_payload.jsonl`.
2. Сгенерируй 50–100 baseline outputs через ту же `meta-llama/Meta-Llama-3-8B-Instruct`.
3. Сохрани в `outputs/ru/baseline_generations.csv`.

### Этап 5. Английское QLoRA-дообучение

1. Используй 300–500 train chunks.
2. Обучи LoRA adapter.
3. Сохрани adapter в `adapters/en_lora_adapter/`.
4. Сгенерируй outputs на тех же prompts, что и baseline.
5. Сохрани в `outputs/en/finetuned_generations.csv`.

### Этап 6. Русское QLoRA-дообучение

1. Используй 300–500 train chunks.
2. Обучи LoRA adapter на той же base model.
3. Сохрани adapter в `adapters/ru_lora_adapter/`.
4. Сгенерируй outputs на тех же prompts, что и baseline.
5. Сохрани в `outputs/ru/finetuned_generations.csv`.

### Этап 7. Метрики

1. Реализуй отдельную стилометрию для EN и RU.
2. Посчитай метрики для:

   * human/eval fragments;
   * baseline generations;
   * finetuned generations.
3. Сохрани:

   * `outputs/en/stylometric_features.csv`
   * `outputs/ru/stylometric_features.csv`

### Этап 8. Статистика

1. Сравни:

   * human vs baseline;
   * human vs finetuned;
   * baseline vs finetuned.
2. Сохрани:

   * `outputs/en/metric_summary.csv`
   * `outputs/ru/metric_summary.csv`

### Этап 9. Ручная оценка

1. Создай blinded CSV для экспертной оценки.
2. Включи туда по 10 текстов из каждой experimental condition.
3. Не показывай condition в blinded version.
4. Сохрани mapping отдельно.

### Этап 10. Финальная проверка

Проверь, что в репозитории есть:

```text
README.md
requirements.txt
configs/
src/
notebooks/
data/processed/
outputs/
adapters/
```

И что можно воспроизвести эксперимент хотя бы в малом масштабе.

## 17. Что не нужно делать

Не делай следующее без отдельной команды:

* не переводить проект на другую модель;
* не использовать Qwen как основную русскую модель;
* не обучать модель на всём датасете по умолчанию;
* не добавлять сложный web interface;
* не добавлять FastAPI/Streamlit;
* не делать production inference server;
* не оптимизировать prematurely;
* не превращать проект в огромный ML framework;
* не удалять старые файлы без архивации;
* не менять тему исследования.

## 18. Ожидаемый итог

В результате работы должен получиться аккуратный research repository, который позволяет сказать в диссертации:

> В практической части была проведена серия экспериментов по сравнению базовой модели Llama 3 8B Instruct и её QLoRA-адаптированных вариантов на англоязычном и русскоязычном корпусах текстов жанра horror/thriller. Для оценки использовались автоматические стилометрические признаки, статистическое сравнение распределений и ручная экспертная оценка с использованием шкал жанровой выраженности, напряжения, естественности языка, связности и оригинальности.

Главный результат проекта — не сама модель, а воспроизводимый сравнительный эксперимент.
