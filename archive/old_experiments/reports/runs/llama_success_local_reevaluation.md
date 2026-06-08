# Local Re-evaluation Report

Results directory: `results_success`
Length-normalized recomputation: first `140` words per text

## Files and Sample Sizes
| file | rows |
| --- | --- |
| stylometric_features.csv | 72 |
| rank_biserial_effects.csv | 18 |
| detector_summary.csv | 2 |
| train_removed_by_quality_filter.csv | 829 |
| eval_removed_by_quality_filter.csv | 201 |

## Label Distribution
| label | n |
| --- | --- |
| ai_base | 24 |
| ai_finetuned | 24 |
| human | 24 |

## Length and Surface Metrics
| label | n | mean chars | median chars | mean words | median words | mean TTR | mean fear rate | mean quotes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ai_base | 24 | 794.0 | 797.0 | 141.0 | 140.5 | 0.724 | 0.0157 | 0.96 |
| ai_finetuned | 24 | 793.7 | 799.0 | 142.6 | 143.5 | 0.722 | 0.0155 | 1.00 |
| human | 24 | 1452.8 | 1469.5 | 279.5 | 280.0 | 0.578 | 0.0075 | 7.79 |

## Artifact and Cliche Rates
| label | loose artifact rows | real artifact rows | cliche rows | loose hits | real hits | top cliches |
| --- | --- | --- | --- | --- | --- | --- |
| ai_base | 16.7% | 0.0% | 58.3% | LAST:3, FIRST:1 | - | air grew thick:4, like living things:3, whispered my name:3, heavy with the scent:3 |
| ai_finetuned | 20.8% | 0.0% | 25.0% | LAST:3, FIRST:2 | - | heavy with the scent:3, unseen presence:2, couldn't shake the feeling:2, air grew thick:1 |
| human | 0.0% | 0.0% | 0.0% | - | - | - |

## Quality Filter Removals
| split | removed rows | unique stories | artifact flag rate | repeated rows | top markers |
| --- | --- | --- | --- | --- | --- |
| train | 829 | 674 | 98.4% | 2 | FIRST:411, LAST:301, FIRST|LAST:101, <empty>:13, DOWNLOADING|http://|FIRST:1 |
| eval | 201 | 156 | 98.0% | 2 | FIRST:93, LAST:65, FIRST|LAST:37, <empty>:4, http://:1 |

## Human vs AI Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | n_chars | 1.000 | 1.000 |
| human_vs_ai_base | n_words | 1.000 | 1.000 |
| human_vs_ai_base | type_token_ratio | -0.997 | 0.997 |
| human_vs_ai_base | avg_word_len | -0.903 | 0.903 |
| human_vs_ai_base | quote_count | 0.613 | 0.613 |
| human_vs_ai_base | question_count | 0.599 | 0.599 |
| human_vs_ai_base | fear_word_rate | -0.535 | 0.535 |
| human_vs_ai_base | exclamation_count | 0.458 | 0.458 |
| human_vs_ai_base | ellipsis_count | -0.167 | 0.167 |
| human_vs_ai_finetuned | n_chars | 1.000 | 1.000 |
| human_vs_ai_finetuned | n_words | 1.000 | 1.000 |
| human_vs_ai_finetuned | type_token_ratio | -0.997 | 0.997 |
| human_vs_ai_finetuned | avg_word_len | -0.899 | 0.899 |
| human_vs_ai_finetuned | quote_count | 0.597 | 0.597 |
| human_vs_ai_finetuned | question_count | 0.509 | 0.509 |
| human_vs_ai_finetuned | exclamation_count | 0.458 | 0.458 |
| human_vs_ai_finetuned | fear_word_rate | -0.434 | 0.434 |
| human_vs_ai_finetuned | ellipsis_count | -0.083 | 0.083 |

## Length-normalized Effects
| comparison | feature | rank biserial | abs effect |
| --- | --- | --- | --- |
| human_vs_ai_base | avg_word_len | -0.872 | 0.872 |
| human_vs_ai_base | type_token_ratio | -0.682 | 0.682 |
| human_vs_ai_base | fear_word_rate | -0.585 | 0.585 |
| human_vs_ai_base | exclamation_count | 0.333 | 0.333 |
| human_vs_ai_base | question_count | 0.323 | 0.323 |
| human_vs_ai_base | quote_count | 0.306 | 0.306 |
| human_vs_ai_base | ellipsis_count | -0.167 | 0.167 |
| human_vs_ai_finetuned | avg_word_len | -0.819 | 0.819 |
| human_vs_ai_finetuned | type_token_ratio | -0.710 | 0.710 |
| human_vs_ai_finetuned | fear_word_rate | -0.524 | 0.524 |
| human_vs_ai_finetuned | exclamation_count | 0.333 | 0.333 |
| human_vs_ai_finetuned | quote_count | 0.292 | 0.292 |
| human_vs_ai_finetuned | question_count | 0.203 | 0.203 |
| human_vs_ai_finetuned | ellipsis_count | -0.083 | 0.083 |

## Detector Summary
| scenario | accuracy | precision | recall | f1 | roc_auc | accuracy 95% CI | inferred confusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human vs base AI | 0.467 | 0.444 | 0.571 | 0.500 | 0.375 | 24.8%-69.9% | TP=4, FP=5, FN=3, TN=3, n=15 |
| human vs fine-tuned AI | 0.467 | 0.455 | 0.714 | 0.556 | 0.429 | 24.8%-69.9% | TP=5, FP=6, FN=2, TN=2, n=15 |

## Practical Reading
- Human texts are much longer than AI texts: median `280.0` vs AI minimum median `140.5` words. Length is a dominant confounder.
- The quality filter is likely over-removing normal prose because `FIRST` and `LAST` are treated as artifacts.
- Detector metrics are smoke-run diagnostics, not stable evidence, unless the inferred test size is comfortably large.
- For the next run, prioritize a stricter artifact marker list, length-matched generation, and a larger eval set.

