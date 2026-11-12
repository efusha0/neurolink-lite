# Results

Binary left/right motor imagery, chance = 50 %. Accuracy is mean +/- std over
folds (within-subject: subject x fold; cross-subject: subject-grouped folds).
Regenerated automatically; do not edit by hand.

| Model | Protocol | Subjects | Test epochs | Accuracy (%) | Median | Min-Max | Pooled | Subjects >= 70 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| csp_lda | cross-subject | 105 | 4725 | 62.1 +/- 1.8 | 61.7 | 60-65 | 62.1 | 0/105 |
| csp_lda_motor | cross-subject | 105 | 4725 | 62.3 +/- 1.3 | 62.0 | 61-64 | 62.3 | 0/105 |
| eegnet | cross-subject | 105 | 4725 | 65.0 +/- 1.8 | 64.4 | 63-68 | 65.0 | 0/105 |
| csp_lda | within-subject | 105 | 4725 | 59.9 +/- 20.4 | 55.6 | 0-100 | 59.9 | 24/105 |
| csp_lda_motor | within-subject | 105 | 4725 | 61.3 +/- 21.7 | 66.7 | 11-100 | 61.3 | 31/105 |
| eegnet | within-subject | 105 | 4725 | 59.0 +/- 18.8 | 55.6 | 0-100 | 59.0 | 22/105 |

## csp_lda / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1467 | 917 |
| right | 873 | 1468 |

## csp_lda_motor / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1431 | 953 |
| right | 829 | 1512 |

## eegnet / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1670 | 714 |
| right | 939 | 1402 |

## csp_lda / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1479 | 905 |
| right | 991 | 1350 |

## csp_lda_motor / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1495 | 889 |
| right | 940 | 1401 |

## eegnet / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1388 | 996 |
| right | 941 | 1400 |
