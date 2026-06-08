# Отчёт по первому smoke-прогону эксперимента

Этот файл подготовлен как краткое техническое резюме первого запуска Colab-эксперимента. Его можно отправить в ChatGPT или использовать как основу для текстового описания результатов.

## 1. Цель прогона

Цель первого прогона состояла не в получении финального научного результата, а в проверке полного pipeline:

1. загрузка подготовленного payload;
2. генерация текстов базовой Llama 3 8B;
3. QLoRA fine-tuning модели на horror-фрагментах;
4. генерация текстов fine-tuned моделью;
5. стилометрическое сравнение human / base AI / fine-tuned AI;
6. обучение supervised detector для пар `human vs base AI` и `human vs fine-tuned AI`.

Эксперимент является адаптацией идеи статьи *When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text* на англоязычный корпус horror / creepypasta stories.

## 2. Корпус и подготовка данных

Исходный корпус был подготовлен локально из датасета creepypasta stories. Тексты были очищены, разделены на train/eval по историям и нарезаны на фрагменты.

Параметры подготовленного корпуса:

| Показатель | Значение |
|---|---:|
| Всего историй | 3446 |
| Train stories | 2756 |
| Eval stories | 690 |
| Full train chunks | 24201 |
| Full eval chunks | 5787 |
| Удалено eval chunks из-за exact train overlap | 18 |
| Payload train chunks | 2000 |
| Payload eval chunks | 500 |
| Story hash overlap train/eval | 0 |
| Chunk hash overlap train/eval | 0 |

Фрагменты в payload имеют длину примерно 700-1500 символов. Для первого smoke-прогона использовалась малая выборка.

## 3. Параметры первого запуска

По файлам результата видно, что запуск был выполнен в smoke-режиме:

| Параметр | Значение |
|---|---:|
| Train examples для SFT | 80 |
| Eval prompts / generations | 24 |
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Fine-tuning method | QLoRA |
| LoRA trainable params | 167,772,160 |
| All params | 8,198,033,408 |
| Trainable share | 2.0465% |

Важно: первый прогон использовал более тяжёлые параметры fine-tuning, близкие к статье (`LORA_R=64`, `NUM_EPOCHS=5`, learning rate около `2e-4`). Это оказалось слишком агрессивно для маленькой smoke-выборки из 80 примеров.

## 4. Результат base generation

Файл: `experiment_1/data/ai_base_generations.csv`

| Показатель | Значение |
|---|---:|
| Количество генераций | 24 |
| Минимальная длина | 434 символа |
| Медианная длина | 1029 символов |
| Средняя длина | 1016 символов |
| Максимальная длина | 1490 символов |
| Медианное число слов | 187 |

Base generation прошла корректно: тексты являются связными англоязычными horror-фрагментами без служебных токенов и явного мусора.

Пример base AI generation:

```text
As I stood in the doorway of our new home, Gavin's eyes darted around the dimly lit foyer, the scent of stale air and forgotten dreams clinging to the walls. My wife, Emily, whispered her approval, her voice barely audible over the creaking of the old wooden floorboards. The realtor, a woman with a saccharine smile, hovered nearby, her eyes fixed on the certificate of ownership still clutched in my hand. Slightly, I felt a presence behind me. I spun around, but there was no one. The doll on the nearby shelf, its porcelain face frozen in a perpetual smile, seemed to be watching us...
```

Промежуточный вывод: базовая Llama 3 8B в данном setup способна генерировать тематически подходящие horror-фрагменты по автоматически сформированным descriptions.

## 5. Результат fine-tuned generation

Файл: `experiment_1/data/ai_finetuned_generations.csv`

| Показатель | Значение |
|---|---:|
| Количество генераций | 24 |
| Минимальная длина | 3246 символов |
| Медианная длина | 3672.5 символов |
| Средняя длина | 3645 символов |
| Максимальная длина | 3973 символов |
| Медианное число слов | 42 |

Fine-tuned generation в первом прогоне оказалась некорректной. Вместо связного английского horror-текста модель начала генерировать артефактные последовательности, повторяющиеся технические токены и фрагменты разных языков.

Пример проблемной fine-tuned generation:

```text
I.GraphicsUnit.GraphicsUnit/***/ Hlav.GraphicsUnit.GraphicsUnit긔​ -Compatible.GraphicsUnit.GraphicsUnit.GraphicsUnit.GraphicsUnit'gcاسطة-FIRSTgeç.GraphicsUnit.GraphicsUnitisay.GraphicsUnit.GraphicsUnit.GraphicsUnit href.GraphicsUnit.GraphicsUnit.GraphicsUnit.GraphicsUnit...
```

Характерные признаки сбоя:

- повторяющиеся технические строки вроде `GraphicsUnit`;
- фрагменты `#echo`, `Compatible`, `LAST`, `FIRST`;
- смешение нерелевантных языковых фрагментов;
- очень длинные строки с малым количеством нормальных слов;
- отсутствие связной horror-прозы.

Промежуточный вывод: QLoRA adapter первого прогона испортил генеративное поведение модели. Поэтому результаты fine-tuned generation нельзя считать валидными для проверки основной гипотезы.

## 6. Стилометрические признаки

Файл: `experiment_1/outputs/stylometric_features.csv`

Средние значения по группам:

| Label | n_chars | n_words | type_token_ratio | avg_word_len | fear_word_rate | quote_count |
|---|---:|---:|---:|---:|---:|---:|
| human | 1443.042 | 273.417 | 0.589 | 4.124 | 0.009 | 6.042 |
| ai_base | 1016.208 | 184.250 | 0.630 | 4.341 | 0.020 | 1.250 |
| ai_finetuned | 3644.958 | 291.042 | 0.192 | 10.128 | 0.000 | 0.250 |

Интерпретация:

- Base AI тексты отличаются от human по длине, числу слов, кавычкам и частоте horror-лексики, но остаются нормальными текстами.
- Fine-tuned AI тексты резко выбиваются по `n_chars`, `type_token_ratio`, `avg_word_len` и `fear_word_rate`.
- Особенно подозрительны `avg_word_len=10.128` и `type_token_ratio=0.192`: это соответствует повторяющимся артефактным токенам, а не нормальной художественной прозе.

## 7. Rank-biserial effects

Файл: `experiment_1/outputs/rank_biserial_effects.csv`

Наиболее сильные различия:

### Human vs base AI

| Feature | Rank-biserial | Abs effect | p-value |
|---|---:|---:|---:|
| n_words | 0.979 | 0.979 | 6.42e-09 |
| n_chars | 0.868 | 0.868 | 2.67e-07 |
| fear_word_rate | -0.569 | 0.569 | 7.47e-04 |
| type_token_ratio | -0.547 | 0.547 | 1.21e-03 |
| quote_count | 0.524 | 0.524 | 1.15e-03 |

### Human vs fine-tuned AI

| Feature | Rank-biserial | Abs effect | p-value |
|---|---:|---:|---:|
| n_chars | -1.000 | 1.000 | 3.05e-09 |
| type_token_ratio | 1.000 | 1.000 | 3.06e-09 |
| avg_word_len | -1.000 | 1.000 | 3.06e-09 |
| ellipsis_count | -1.000 | 1.000 | 2.21e-10 |
| fear_word_rate | 0.917 | 0.917 | 3.15e-09 |

Интерпретация:

- Fine-tuned texts не стали ближе к human.
- Напротив, они стали радикально отличаться от human по нескольким признакам.
- Rank-biserial effects около `1.0` указывают не на успех fine-tuning, а на сильный артефактный сдвиг распределения.

## 8. Detector results

Файл: `experiment_1/outputs/detector_summary.csv`

| Scenario | Loss | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| human vs base AI | 2.365 | 0.400 | 0.333 | 0.286 | 0.308 | 0.411 |
| human vs fine-tuned AI | 0.708 | 0.867 | 0.778 | 1.000 | 0.875 | 0.875 |

На первый взгляд detector лучше отличает fine-tuned AI от human, чем base AI от human. Однако это не является содержательным научным результатом, потому что fine-tuned AI тексты сломаны и содержат очевидные артефакты. Детектор, вероятно, распознаёт не “AI-style”, а ненормальное распределение токенов и технический мусор.

## 9. Основной вывод по первому прогону

Первый прогон следует рассматривать как диагностический smoke run.

Что подтвердилось:

- payload корректно загружается;
- base Llama 3 8B успешно генерирует связные horror-фрагменты;
- QLoRA training технически запускается;
- downstream-этапы stylometry и detector выполняются;
- результаты сохраняются в CSV.

Что не подтвердилось:

- fine-tuned model не дала пригодных horror-текстов;
- результаты `human vs fine-tuned AI` нельзя использовать для проверки гипотезы статьи;
- текущие detector/stylometry метрики отражают сбой генерации, а не успешную имитацию человеческого стиля.

## 10. Вероятная причина сбоя

Наиболее вероятная причина — слишком агрессивный fine-tuning для малого smoke-набора:

- `LORA_R=64` даёт большой адаптер;
- `NUM_EPOCHS=5` слишком много для 80 SFT-примеров;
- learning rate около `2e-4` мог быть слишком высоким;
- маленький SFT-набор повышает риск переобучения или дестабилизации генерации.

## 11. Что изменить в следующем прогоне

Для следующего smoke run рекомендуется использовать более мягкие параметры:

```python
SMOKE_TEST = True
NUM_EPOCHS = 1
LORA_R = 16
LEARNING_RATE = 5e-5
MAX_NEW_TOKENS = 250
SMOKE_TRAIN_SAMPLES = 80
SMOKE_EVAL_SAMPLES = 24
```

Ожидаемый критерий успешности следующего smoke run:

1. `ai_base_generations.csv` содержит нормальные English horror texts;
2. `ai_finetuned_generations.csv` тоже содержит нормальные English horror texts;
3. fine-tuned тексты не содержат `GraphicsUnit`, `#echo`, HTML/CSS/code-like фрагментов или смешения случайных языков;
4. только после этого можно интерпретировать stylometry и detector metrics.

## 12. Как этот прогон описывать в тексте работы

Этот прогон не следует описывать как финальный результат эксперимента. Корректная формулировка:

> Первый smoke-прогон подтвердил работоспособность общего pipeline: подготовленный payload был успешно загружен, базовая Llama 3 8B сгенерировала связные horror-фрагменты, QLoRA fine-tuning был технически выполнен, а модули стилометрического анализа и detector evaluation отработали до конца. Однако fine-tuned модель в этом запуске начала генерировать артефактные последовательности, не являющиеся связным английским текстом. Поэтому полученные detector и stylometry metrics были признаны диагностическими и не использовались как подтверждение основной гипотезы. Для следующего запуска параметры fine-tuning были смягчены: уменьшены LoRA rank, learning rate, число эпох и максимальная длина генерации.

## 13. Короткое резюме для ChatGPT

Если нужно попросить ChatGPT написать связный текстовый анализ, можно использовать такой запрос:

```text
Ниже приведён технический отчёт по первому smoke-прогону эксперимента. Напиши по нему связное описание результатов для исследовательской работы. Важно: не представляй этот прогон как успешное подтверждение гипотезы. Подчеркни, что base generation прошла корректно, но fine-tuned generation дала артефактный текст, поэтому detector/stylometry results являются диагностическими. Заверши выводом о необходимости повторного запуска с более мягкими параметрами QLoRA.
```

