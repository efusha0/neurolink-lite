# Results

Binary left/right motor imagery, chance = 50 %. Every statistic except the last
column is over per-subject accuracies (each subject's own test epochs, pooled
across folds), so the spread shows inter-subject variability. "Fold std" is the
std over folds (within-subject: subject x fold, 9 test epochs each, hence large;
cross-subject: the 5 subject-grouped folds).
Regenerated automatically; do not edit by hand.

| Model | Protocol | Subjects | Test epochs | Accuracy (%), mean +/- subject std | Median | Min-Max | Pooled | Subjects >= 70 % | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| csp_lda | cross-subject | 105 | 4725 | 62.1 +/- 12.6 | 57.8 | 38-93 | 62.1 | 31/105 | 1.8 |
| csp_lda_motor | cross-subject | 105 | 4725 | 62.3 +/- 12.8 | 60.0 | 36-93 | 62.3 | 29/105 | 1.3 |
| eegnet | cross-subject | 105 | 4725 | 65.0 +/- 13.2 | 62.2 | 29-96 | 65.0 | 35/105 | 1.8 |
| csp_lda | within-subject | 105 | 4725 | 59.9 +/- 15.5 | 57.8 | 33-98 | 59.9 | 24/105 | 20.4 |
| csp_lda_motor | within-subject | 105 | 4725 | 61.3 +/- 16.5 | 60.0 | 27-100 | 61.3 | 31/105 | 21.7 |
| eegnet | within-subject | 105 | 4725 | 59.0 +/- 14.1 | 57.8 | 31-98 | 59.0 | 22/105 | 18.8 |

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
