# Local Re-evaluation Report

Results directory: `experiment_1/outputs`
Length-normalized recomputation: first `187` words per text

## Files and Sample Sizes
| file | rows |
| --- | --- |
| stylometric_features.csv | 72 |
| rank_biserial_effects.csv | 18 |
| detector_summary.csv | 2 |
| train_removed_by_quality_filter.csv | 0 |
| eval_removed_by_quality_filter.csv | 0 |

## Label Distribution
| label | n |
| --- | --- |
| ai_base | 24 |
| ai_finetuned | 24 |
| human | 24 |

## Length and Surface Metrics
| label | n | mean chars | median chars | mean words | median words | mean TTR | mean fear rate | mean quotes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ai_base | 24 | 1016.2 | 1029.0 | 184.2 | 187.0 | 0.630 | 0.0196 | 1.25 |
| ai_finetuned | 24 | 3645.0 | 3672.5 | 291.0 | 292.5 | 0.192 | 0.0000 | 0.25 |
| human | 24 | 1443.0 | 1453.0 | 273.4 | 275.0 | 0.589 | 0.0087 | 6.04 |

## Artifact and Cliche Rates
| label | loose artifact rows | real artifact rows | cliche rows | loose hits | real hits | top cliches |
| --- | --- | --- | --- | --- | --- | --- |
| ai_base | 8.3% | 0.0% | 41.7% | LAST:1, FIRST:1 | - | heavy with the scent:4, air grew thick:3, darkness seemed to:2, the darkness seemed:2 |
| ai_finetuned | 100.0% | 100.0% | 0.0% | GraphicsUnit:24, #echo:24, Compatible:24, وینت:24 | GraphicsUnit:24, #echo:24, Compatible:24, وینت:24 | - |
| human | 37.5% | 0.0% | 0.0% | FIRST:6, LAST:3 | - | - |

## Human vs AI Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | n_words | 0.979 | 0.979 |
| human_vs_ai_base | n_chars | 0.868 | 0.868 |
| human_vs_ai_base | fear_word_rate | -0.569 | 0.569 |
| human_vs_ai_base | type_token_ratio | -0.547 | 0.547 |
| human_vs_ai_base | quote_count | 0.524 | 0.524 |
| human_vs_ai_base | avg_word_len | -0.514 | 0.514 |
| human_vs_ai_base | question_count | 0.335 | 0.335 |
| human_vs_ai_base | exclamation_count | 0.250 | 0.250 |
| human_vs_ai_base | ellipsis_count | 0.000 | 0.000 |
| human_vs_ai_finetuned | n_chars | -1.000 | 1.000 |
| human_vs_ai_finetuned | type_token_ratio | 1.000 | 1.000 |
| human_vs_ai_finetuned | avg_word_len | -1.000 | 1.000 |
| human_vs_ai_finetuned | ellipsis_count | -1.000 | 1.000 |
| human_vs_ai_finetuned | fear_word_rate | 0.917 | 0.917 |
| human_vs_ai_finetuned | exclamation_count | -0.892 | 0.892 |
| human_vs_ai_finetuned | quote_count | 0.688 | 0.688 |
| human_vs_ai_finetuned | n_words | -0.583 | 0.583 |
| human_vs_ai_finetuned | question_count | 0.500 | 0.500 |

## Length-normalized Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | avg_word_len | -0.533 | 0.533 |
| human_vs_ai_base | fear_word_rate | -0.490 | 0.490 |
| human_vs_ai_base | quote_count | 0.314 | 0.314 |
| human_vs_ai_base | question_count | 0.170 | 0.170 |
| human_vs_ai_base | exclamation_count | 0.125 | 0.125 |
| human_vs_ai_base | type_token_ratio | -0.087 | 0.087 |
| human_vs_ai_base | ellipsis_count | 0.000 | 0.000 |
| human_vs_ai_finetuned | type_token_ratio | 1.000 | 1.000 |
| human_vs_ai_finetuned | avg_word_len | -1.000 | 1.000 |
| human_vs_ai_finetuned | ellipsis_count | -1.000 | 1.000 |
| human_vs_ai_finetuned | fear_word_rate | 0.875 | 0.875 |
| human_vs_ai_finetuned | exclamation_count | -0.816 | 0.816 |
| human_vs_ai_finetuned | quote_count | 0.446 | 0.446 |
| human_vs_ai_finetuned | question_count | 0.333 | 0.333 |

## Detector Summary
| scenario | accuracy | precision | recall | f1 | roc_auc | accuracy 95% CI | inferred confusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human vs base AI | 0.400 | 0.333 | 0.286 | 0.308 | 0.411 | 19.8%-64.3% | TP=2, FP=4, FN=5, TN=4, n=15 |
| human vs fine-tuned AI | 0.867 | 0.778 | 1.000 | 0.875 | 0.875 | 62.1%-96.3% | TP=7, FP=2, FN=0, TN=6, n=15 |

## Practical Reading
- Human texts are much longer than AI texts: median `275.0` vs AI minimum median `187.0` words. Length is a dominant confounder.
- Detector metrics are smoke-run diagnostics, not stable evidence, unless the inferred test size is comfortably large.
- For the next run, prioritize a stricter artifact marker list, length-matched generation, and a larger eval set.

