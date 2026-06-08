# Отчёт по experiment_2

Этот отчёт фиксирует результаты второго smoke-прогона. Основная цель проверки — понять, исчезла ли проблема артефактной fine-tuned генерации после изменения параметров запуска.

## 1. Состав результатов

Папка `experiment_2/` содержит:

- `data/ai_base_generations.csv`
- `data/ai_finetuned_generations.csv`
- `outputs/stylometric_features.csv`
- `outputs/rank_biserial_effects.csv`
- `outputs/detector_summary.csv`
- `models/horror_llama3_qlora_adapter_final/adapter_model.safetensors`

Размер папки: около `3.2G`. Из-за model adapter её не следует коммитить в git целиком.

## 2. Размеры выборок

| Файл | Строк |
|---|---:|
| `ai_base_generations.csv` | 24 |
| `ai_finetuned_generations.csv` | 24 |
| `stylometric_features.csv` | 72 |

Как и в первом запуске, это smoke-режим с 24 eval prompts.

## 3. Base generation

Base generation выглядит корректно: модель генерирует связные англоязычные horror-фрагменты.

Статистика base AI:

| Показатель | Значение |
|---|---:|
| min chars | 425 |
| median chars | 993 |
| mean chars | 926 |
| max chars | 1170 |
| median words | 175.5 |
| mean words | 166.7 |

Пример:

```text
As I stepped into the foyer, the realtor's forced smile still lingered in my mind, like the faint scent of fresh paint. Gavin's eyes, however, were already elsewhere. They wandered the room, as if searching for something I couldn't see. My wife, Emma, held my hand, her grip tightening slightly as we took in the open-plan living area. The doll on the coffee table seemed out of place, its glassy stare following us as we moved...
```

Вывод: base Llama 3 8B работает корректно и генерирует пригодные horror-тексты.

## 4. Fine-tuned generation

Fine-tuned generation снова невалидна. Все 24 fine-tuned генерации содержат технические/мусорные токены.

Статистика fine-tuned AI:

| Показатель | Значение |
|---|---:|
| min chars | 1224 |
| median chars | 1508.5 |
| mean chars | 1496 |
| max chars | 1722 |
| median words | 50 |
| mean words | 49 |

Артефакты найдены во всех fine-tuned строках:

| Маркер | Количество строк из 24 |
|---|---:|
| `GraphicsUnit` | 24 |
| `#echo` | 24 |
| `Compatible` | 24 |
| `다운받기` | 24 |
| `وینت` | 24 |
| non-ASCII symbols | 24 |

Пример:

```text
G#echoampie-License-LAST.AnchorStyles#echo ensure/ namespaceummings,…#echosilverوینت.GraphicsUnit.GraphicsUnit.GraphicsUnit.GraphicsUnit.GraphicsUnit.GraphicsUnit/Object#echo Increamedi 다운받기geçاسطة...
```

Вывод: несмотря на изменение запуска, fine-tuned модель продолжает генерировать не английскую прозу, а артефактные последовательности.

## 5. Стилометрия

Средние значения:

| Label | n_chars | n_words | type_token_ratio | avg_word_len | fear_word_rate | quote_count | ellipsis_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 1443.042 | 273.417 | 0.589 | 4.124 | 0.009 | 6.042 | 0.000 |
| ai_base | 926.375 | 166.708 | 0.650 | 4.371 | 0.017 | 1.917 | 0.333 |
| ai_finetuned | 1496.208 | 134.167 | 0.505 | 8.905 | 0.000 | 0.333 | 2.333 |

Интерпретация:

- Fine-tuned тексты приблизились к human по `n_chars`, но это ложное сходство: длина обеспечена техническим мусором.
- `avg_word_len=8.905` и `fear_word_rate=0.000` указывают на отсутствие нормальной horror-лексики.
- Повышенный `ellipsis_count` и низкий `quote_count` также отражают артефактную структуру генераций.

## 6. Rank-biserial effects

Наиболее сильные различия:

### Human vs base AI

| Feature | Rank-biserial | Abs effect |
|---|---:|---:|
| n_chars | 1.000 | 1.000 |
| n_words | 1.000 | 1.000 |
| type_token_ratio | -0.741 | 0.741 |
| avg_word_len | -0.545 | 0.545 |
| fear_word_rate | -0.444 | 0.444 |

### Human vs fine-tuned AI

| Feature | Rank-biserial | Abs effect |
|---|---:|---:|
| n_words | 1.000 | 1.000 |
| avg_word_len | -1.000 | 1.000 |
| fear_word_rate | 0.917 | 0.917 |
| ellipsis_count | -0.917 | 0.917 |
| exclamation_count | -0.891 | 0.891 |

Интерпретация:

- Fine-tuned тексты остаются резко отличимыми от human.
- Сильные эффекты связаны не с литературным стилем, а с поломкой генерации.

## 7. Detector results

| Scenario | Loss | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| human vs base AI | 2.524 | 0.400 | 0.250 | 0.143 | 0.182 | 0.411 |
| human vs fine-tuned AI | 0.784 | 0.867 | 0.778 | 1.000 | 0.875 | 0.893 |

Detector снова лучше отличает `fine-tuned AI` от human, но это не является научным подтверждением гипотезы. Fine-tuned тексты содержат явные артефакты, поэтому detector учится распознавать сбой генерации.

## 8. Сравнение с experiment_1

Experiment 2 немного уменьшил длину fine-tuned мусора:

| Показатель | Experiment 1 fine-tuned | Experiment 2 fine-tuned |
|---|---:|---:|
| median chars | 3672.5 | 1508.5 |
| median words | 42 | 50 |
| artifact markers | есть | есть во всех строках |

Проблема не исчезла. Она стала короче по длине генерации, но качественно осталась той же.

## 9. Главный вывод

`experiment_2` не является валидным экспериментальным результатом для проверки гипотезы статьи.

Что подтверждено:

- base generation работает стабильно;
- весь pipeline снова доходит до конца;
- detector/stylometry technically run.

Что не подтверждено:

- fine-tuned generation не стала нормальной английской horror-прозой;
- fine-tuned модель не стала ближе к human;
- detector results нельзя интерпретировать как содержательное сравнение human vs AI.

## 10. Что делать дальше

Перед следующим запуском нужно исправлять не detector и не stylometry, а сам fine-tuning/generation setup.

Рекомендуемые варианты:

1. Проверить, что adapter применяется к той же модели и tokenizer, на которых он обучался.
2. Уменьшить агрессивность обучения ещё сильнее:

```python
NUM_EPOCHS = 1
LORA_R = 8
LEARNING_RATE = 1e-5
MAX_NEW_TOKENS = 180
SMOKE_TRAIN_SAMPLES = 300
```

3. Генерировать fine-tuned samples через заново загруженную модель:

```python
fresh_base_model = AutoModelForCausalLM.from_pretrained(...)
ft_model = PeftModel.from_pretrained(fresh_base_model, adapter_path)
```

4. Добавить контрольную генерацию сразу после training на 1-2 prompts перед запуском detector.
5. Если артефакты сохраняются даже после fresh reload, проверять формат SFT-примеров и параметры quantization/dtype.

## 11. Формулировка для исследовательского текста

Experiment 2 следует описывать как повторный диагностический запуск:

> Во втором smoke-прогоне были повторно проверены все этапы pipeline после изменения параметров запуска. Базовая Llama 3 8B снова сгенерировала связные англоязычные horror-фрагменты. Однако fine-tuned модель продолжила генерировать артефактные последовательности с техническими токенами и нерелевантными символами. Несмотря на то что detector показал высокую accuracy для пары human vs fine-tuned AI, этот результат отражает не успешное обнаружение AI-текста, а распознавание очевидно повреждённых генераций. Поэтому experiment 2 был признан диагностическим и не использовался как финальное подтверждение или опровержение гипотезы.
