# Local Re-evaluation Report

Results directory: `experiment_2/outputs`
Length-normalized recomputation: first `135` words per text

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
| ai_base | 24 | 926.4 | 993.0 | 166.7 | 176.5 | 0.650 | 0.0169 | 1.92 |
| ai_finetuned | 24 | 1496.2 | 1508.5 | 134.2 | 135.0 | 0.505 | 0.0000 | 0.33 |
| human | 24 | 1443.0 | 1453.0 | 273.4 | 275.0 | 0.589 | 0.0087 | 6.04 |

## Artifact and Cliche Rates
| label | loose artifact rows | real artifact rows | cliche rows | loose hits | real hits | top cliches |
| --- | --- | --- | --- | --- | --- | --- |
| ai_base | 33.3% | 0.0% | 37.5% | LAST:4, FIRST:4 | - | couldn't shake the feeling:4, darkness seemed to:2, the darkness seemed:2, like living things:2 |
| ai_finetuned | 100.0% | 100.0% | 0.0% | GraphicsUnit:24, #echo:24, Compatible:24, 다운받기:24 | GraphicsUnit:24, #echo:24, Compatible:24, 다운받기:24 | - |
| human | 37.5% | 0.0% | 0.0% | FIRST:6, LAST:3 | - | - |

## Human vs AI Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | n_chars | 1.000 | 1.000 |
| human_vs_ai_base | n_words | 1.000 | 1.000 |
| human_vs_ai_base | type_token_ratio | -0.741 | 0.741 |
| human_vs_ai_base | avg_word_len | -0.545 | 0.545 |
| human_vs_ai_base | fear_word_rate | -0.444 | 0.444 |
| human_vs_ai_base | quote_count | 0.411 | 0.411 |
| human_vs_ai_base | question_count | 0.406 | 0.406 |
| human_vs_ai_base | exclamation_count | 0.250 | 0.250 |
| human_vs_ai_base | ellipsis_count | -0.250 | 0.250 |
| human_vs_ai_finetuned | n_words | 1.000 | 1.000 |
| human_vs_ai_finetuned | avg_word_len | -1.000 | 1.000 |
| human_vs_ai_finetuned | ellipsis_count | -0.917 | 0.917 |
| human_vs_ai_finetuned | fear_word_rate | 0.917 | 0.917 |
| human_vs_ai_finetuned | exclamation_count | -0.891 | 0.891 |
| human_vs_ai_finetuned | type_token_ratio | 0.788 | 0.788 |
| human_vs_ai_finetuned | quote_count | 0.667 | 0.667 |
| human_vs_ai_finetuned | question_count | 0.500 | 0.500 |
| human_vs_ai_finetuned | n_chars | -0.443 | 0.443 |

## Length-normalized Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | avg_word_len | -0.495 | 0.495 |
| human_vs_ai_base | fear_word_rate | -0.422 | 0.422 |
| human_vs_ai_base | type_token_ratio | -0.214 | 0.214 |
| human_vs_ai_base | question_count | 0.200 | 0.200 |
| human_vs_ai_base | quote_count | 0.137 | 0.137 |
| human_vs_ai_base | exclamation_count | 0.125 | 0.125 |
| human_vs_ai_base | ellipsis_count | -0.042 | 0.042 |
| human_vs_ai_finetuned | avg_word_len | -1.000 | 1.000 |
| human_vs_ai_finetuned | type_token_ratio | 0.972 | 0.972 |
| human_vs_ai_finetuned | exclamation_count | -0.929 | 0.929 |
| human_vs_ai_finetuned | ellipsis_count | -0.917 | 0.917 |
| human_vs_ai_finetuned | fear_word_rate | 0.667 | 0.667 |
| human_vs_ai_finetuned | question_count | 0.292 | 0.292 |
| human_vs_ai_finetuned | quote_count | 0.278 | 0.278 |

## Detector Summary
| scenario | accuracy | precision | recall | f1 | roc_auc | accuracy 95% CI | inferred confusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human vs base AI | 0.400 | 0.250 | 0.143 | 0.182 | 0.411 | 19.8%-64.3% | TP=1, FP=3, FN=6, TN=5, n=15 |
| human vs fine-tuned AI | 0.867 | 0.778 | 1.000 | 0.875 | 0.893 | 62.1%-96.3% | TP=7, FP=2, FN=0, TN=6, n=15 |

## Practical Reading
- Human texts are much longer than AI texts: median `275.0` vs AI minimum median `135.0` words. Length is a dominant confounder.
- Detector metrics are smoke-run diagnostics, not stable evidence, unless the inferred test size is comfortably large.
- For the next run, prioritize a stricter artifact marker list, length-matched generation, and a larger eval set.

