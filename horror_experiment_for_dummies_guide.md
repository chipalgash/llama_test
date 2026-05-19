
# Инструкция for dummies: как повторить эксперимент на хоррор-историях

## Что делает notebook

Notebook повторяет основную логику статьи *When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text*, но переносит её с коротких соцсетевых постов на англоязычные хоррор-истории.

Оригинальный эксперимент строился так: авторы собирали человеческие тексты, генерировали AI-тексты разными стратегиями, дообучали модели через QLoRA, сравнивали human/base/fine-tuned тексты и проверяли, насколько хорошо детекторы отличают AI-текст от человеческого. В статье использовались GPT-4o, GPT-4o-mini, Llama-3-8B-Instruct и Llama-3.2-1B-Instruct; для Llama применялись QLoRA, 4-bit quantization, LoRA rank 64, alpha 16 и 5 эпох обучения. В notebook эта схема адаптирована под horror corpus и фиксируется на `Meta-Llama-3-8B-Instruct`.

## Что именно берём из статьи

Для нашей репликации оставляем не всю статью, а её ключевой дизайн:

- human-written corpus: вместо Twitter/X берём англоязычные creepypasta/horror stories;
- generation strategy: используем аналог `Generate From Topic`, то есть сначала делаем краткое описание фрагмента, затем просим модель написать новый horror fragment по описанию;
- base condition: генерируем тексты базовой `Meta-Llama-3-8B-Instruct`;
- fine-tuned condition: дообучаем ту же Llama 3 8B через QLoRA и снова генерируем по тем же описаниям;
- comparison: считаем стилометрию и обучаем supervised detector для `human vs base AI` и `human vs fine-tuned AI`;
- главный ожидаемый эффект: fine-tuned AI должен быть ближе к human по признакам и труднее для детектора.

## Что тебе нужно подготовить

1. Собери `.txt` файлы с хоррор-историями.
2. Проверь лицензию текстов. Лучше использовать public domain, CC BY, CC BY-SA, CC0 или тексты с разрешением автора.
3. Сохрани все `.txt` в одну папку.
4. Запусти локальный payload-скрипт, чтобы получить `data/horror_experiment_payload.zip`.

Пример:

```bash
python3 scripts/prepare_colab_payload.py
```

На выходе нужен файл:

```text
data/horror_experiment_payload.zip
```

## Если используешь Kaggle-датасет 3500 Popular Creepypastas

Для датасета `thomaskonstantin/3500-popular-creepypastas` добавлен отдельный скрипт:

```bash
python3 scripts/extract_creepypastas.py \
  --input creepypastas.xlsx \
  --output-dir data/creepypasta_stories_txt \
  --overwrite
```

Что делает скрипт:

- принимает Kaggle ZIP, распакованную папку или отдельный XLSX/CSV/JSON/TXT;
- автоматически ищет колонку с текстом истории;
- чистит HTML и лишние пробелы;
- удаляет пустые, слишком короткие и дублирующиеся записи;
- сохраняет каждую историю отдельным `.txt` файлом;
- создаёт `metadata.csv` со списком извлечённых историй.

После запуска получится папка:

```text
data/creepypasta_stories_txt/
├── 0001_story_title.txt
├── 0002_another_story.txt
└── metadata.csv
```

Если скрипт не угадал колонку с текстом, укажи её явно:

```bash
python3 scripts/extract_creepypastas.py \
  --input creepypastas.xlsx \
  --text-column body \
  --title-column story_name \
  --output-dir data/creepypasta_stories_txt \
  --overwrite
```

Эта папка лежит внутри `data/`, поэтому не попадёт в git. Для Colab используй не папку с `.txt`, а готовый payload:

```bash
python3 scripts/prepare_colab_payload.py
```

## Как запустить

1. Открой Google Colab.
2. Нажми `File → Upload notebook`.
3. Загрузи файл `horror_llama3_payload_experiment_colab.ipynb`.
4. В меню выбери `Runtime → Change runtime type`.
5. В поле `Hardware accelerator` выбери `T4 GPU`.
6. Когда notebook попросит файл, загрузи `data/horror_experiment_payload.zip`.
7. Запускай ячейки сверху вниз.

## Где нужно что-то менять

Главный блок называется `CONFIG`.

Эксперимент теперь зафиксирован как English-only + Llama 3 8B:

```python
"GEN_MODEL_NAME": "meta-llama/Meta-Llama-3-8B-Instruct"
"DETECTOR_MODEL_NAME": "openai-community/roberta-base-openai-detector"
"SMOKE_TEST": True
"SMOKE_TRAIN_SAMPLES": 80
"SMOKE_EVAL_SAMPLES": 24
"NUM_EPOCHS": 5
```

Для полного запуска отключи smoke test:

```python
"SMOKE_TEST": False
```

Для Llama 3 8B нужен Hugging Face доступ:

1. Открой страницу `meta-llama/Meta-Llama-3-8B-Instruct` на Hugging Face.
2. Прими лицензию Meta.
3. В Colab добавь `HF_TOKEN` в secrets или запусти ячейку `notebook_login()`.

## Что происходит по шагам

### Шаг 1. Установка библиотек

Notebook устанавливает библиотеки для:
- загрузки моделей;
- QLoRA fine-tuning;
- обработки датасетов;
- обучения детектора;
- расчёта метрик.

### Шаг 2. Загрузка payload

Ты загружаешь `horror_experiment_payload.zip`. Notebook распаковывает уже подготовленные CSV:

```text
train_with_descriptions.csv
eval_with_descriptions.csv
split_metadata.csv
corpus_summary.csv
corpus_preparation_report.md
```

### Шаг 3. Проверка train/eval split

Нарезка текстов и descriptions уже сделаны локально скриптом `scripts/prepare_colab_payload.py`. В notebook проверяется, сколько train/eval chunks загружено и включён ли `SMOKE_TEST`.

По умолчанию:

```bash
python3 scripts/prepare_colab_payload.py --min-chars 700 --max-chars 1500
```

То есть каждый пример будет примерно от 700 до 1500 символов.

### Шаг 4. Загрузка Llama 3 8B

Notebook загружает `meta-llama/Meta-Llama-3-8B-Instruct` в 4-bit режиме. Это требует GPU и Hugging Face доступа к gated Llama model.

### Шаг 5. Генерация base AI

Базовая модель генерирует хоррор-фрагменты до дообучения.

Это условие нужно, чтобы понять, насколько “обычная” модель отличается от человеческих текстов.

### Шаг 6. QLoRA fine-tuning

Модель дообучается на парах:

```text
описание → человеческий хоррор-фрагмент
```

То есть она учится писать в стиле твоего корпуса.

### Шаг 7. Генерация fine-tuned AI

После обучения та же модель снова генерирует тексты по тем же описаниям.

Теперь можно сравнить:

```text
human vs base AI
human vs fine-tuned AI
```

### Шаг 8. Стилометрический анализ

Notebook считает признаки:

- длина текста;
- число слов;
- type-token ratio;
- доля заглавных букв;
- восклицательные знаки;
- вопросительные знаки;
- многоточия;
- кавычки / признаки прямой речи;
- частотность слов страха.

Главная идея: если fine-tuned AI ближе к human по этим признакам, значит дообучение сработало.

### Шаг 9. Детектор

Notebook обучает простой supervised detector:

1. human vs base AI;
2. human vs fine-tuned AI.

Если accuracy/F1 ниже во втором случае, значит fine-tuned тексты труднее отличить от человеческих.

## Какие файлы получатся

После запуска будут созданы:

```text
horror_experiment/
├── data/
│   ├── human_horror_chunks.csv
│   ├── human_with_descriptions.csv
│   ├── ai_base_generations.csv
│   └── ai_finetuned_generations.csv
├── outputs/
│   ├── stylometric_features.csv
│   ├── rank_biserial_effects.csv
│   ├── detector_summary.csv
│   └── mixed_examples_for_manual_review.csv
└── models/
    ├── horror_qlora_adapter_final/
    ├── detector_human_vs_base_ai/
    └── detector_human_vs_finetuned_ai/
```

Самые важные файлы:

- `ai_base_generations.csv` — тексты базовой модели;
- `ai_finetuned_generations.csv` — тексты после дообучения;
- `rank_biserial_effects.csv` — различия между human и AI;
- `detector_summary.csv` — итоговые метрики детектора;
- `mixed_examples_for_manual_review.csv` — примеры для ручного анализа.

## Как интерпретировать результат

Тебе нужны два основных вывода.

### Вывод 1

Если стилометрические различия между human и fine-tuned AI меньше, чем между human и base AI, значит fine-tuning сделал модель ближе к человеческому корпусу.

Смотри файл:

```text
rank_biserial_effects.csv
```

Чем ближе `rank_biserial` к нулю, тем меньше различие.

### Вывод 2

Если детектор хуже отличает fine-tuned AI от human, чем base AI от human, значит эксперимент повторяет основную идею статьи.

Смотри файл:

```text
detector_summary.csv
```

Пример желаемой картины:

```text
human vs base AI          accuracy = 0.90
human vs fine-tuned AI    accuracy = 0.70
```

Это значит, что после fine-tuning AI-текст стал менее обнаружимым.

## Что написать в работе

Можно использовать такую формулировку:

> В практической части была реализована адаптированная репликация эксперимента Dawkins et al. по оценке обнаружимости текстов, созданных базовой и дообученной языковой моделью. В отличие от оригинального исследования, материалом стали фрагменты хоррор-историй. Корпус человеческих текстов был очищен и сегментирован на фрагменты сопоставимой длины. Для каждого фрагмента автоматически формировалось краткое генеративное описание, после чего базовая instruction-модель создавала синтетические хоррор-фрагменты. Затем модель была дообучена методом QLoRA на парах “описание — человеческий фрагмент”, после чего генерация была повторена. Полученные тексты сравнивались со стилометрическими признаками человеческого корпуса, а также оценивались с помощью supervised PLM-детектора.

## Важные ограничения

Это не полная копия статьи, а адаптированная репликация.

Отличия:
- вместо Twitter/X используются художественные хоррор-фрагменты;
- вместо политических topic/stance используется horror description;
- off-the-shelf детекторы для художественного horror-домена ненадёжны;
- основной детектор обучается внутри эксперимента;
- качество зависит от лицензий, объёма и чистоты корпуса.

## Частые ошибки

### Ошибка: CUDA out of memory

Что делать:
- включить `SMOKE_TEST`;
- уменьшить `SMOKE_TRAIN_SAMPLES` и `SMOKE_EVAL_SAMPLES`;
- поставить `LORA_R = 16`;
- уменьшить `MAX_NEW_TOKENS`;
- если нужно строго следовать статье, оставить Llama 3 8B; если нужно только отладить код, временно заменить модель на меньшую.

### Ошибка: нет GPU

Что делать:
- `Runtime → Change runtime type → T4 GPU`.

### Ошибка с Llama access

Llama-модели требуют принятия лицензии на Hugging Face. Для текущего эксперимента не заменяй модель на Qwen: нужно принять лицензию `meta-llama/Meta-Llama-3-8B-Instruct` и авторизоваться через `HF_TOKEN` или `notebook_login()`.

### Генерации слишком короткие или плохие

Увеличь:

```python
"MAX_NEW_TOKENS": 500
```

И проверь качество исходных `.txt`.

### Модель копирует исходные тексты

Это риск при маленьком корпусе. Что делать:
- увеличить корпус;
- уменьшить число эпох;
- вручную проверить `mixed_examples_for_manual_review.csv`;
- добавить проверку на near-duplicate.
