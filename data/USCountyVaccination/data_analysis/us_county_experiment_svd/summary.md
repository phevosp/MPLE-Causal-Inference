# US County Experiment Start-Date SVD Analysis

This summary analyzes the saved experiment transition panels `z` and `x` after applying the standard trimmed support rule, the `2w` intervention lag, dense-suffix support selection, and then the requested start-date slicing.

## Files

- Intervention summaries: `intervention_svd_summary.csv`
- Intervention spectra: `intervention_svd_spectra.csv`
- Outcome summaries: `outcome_svd_summary.csv`
- Outcome spectra: `outcome_svd_spectra.csv`

## Intervention Highlights

- `complete_cov_ge_20` from `2020-03-01` (resolved `2020-03-01`): `weak` low-rank evidence after row/column centering; top component energy `0.6003`, `95%` rank `11`.
- `complete_cov_ge_20` from `2020-06-07` (resolved `2020-06-07`): `weak` low-rank evidence after row/column centering; top component energy `0.5875`, `95%` rank `11`.
- `complete_cov_ge_20` from `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.5710`, `95%` rank `11`.
- `complete_cov_ge_30` from `2020-03-01` (resolved `2020-03-01`): `weak` low-rank evidence after row/column centering; top component energy `0.5429`, `95%` rank `12`.
- `complete_cov_ge_30` from `2020-06-07` (resolved `2020-06-07`): `weak` low-rank evidence after row/column centering; top component energy `0.5314`, `95%` rank `13`.
- `complete_cov_ge_30` from `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.5176`, `95%` rank `13`.
- `complete_cov_ge_40` from `2020-03-01` (resolved `2020-03-01`): `moderate` low-rank evidence after row/column centering; top component energy `0.4714`, `95%` rank `11`.
- `complete_cov_ge_40` from `2020-06-07` (resolved `2020-06-07`): `moderate` low-rank evidence after row/column centering; top component energy `0.4445`, `95%` rank `12`.
- `complete_cov_ge_40` from `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.4084`, `95%` rank `12`.

## Outcome Highlights

- Support from `complete_cov_ge_20`, start `2020-03-01` (resolved `2020-03-01`): `weak` low-rank evidence after row/column centering; top component energy `0.0542`, `95%` rank `97`.
- Support from `complete_cov_ge_20`, start `2020-06-07` (resolved `2020-06-07`): `weak` low-rank evidence after row/column centering; top component energy `0.0588`, `95%` rank `89`.
- Support from `complete_cov_ge_20`, start `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.0519`, `95%` rank `78`.
- Support from `complete_cov_ge_30`, start `2020-03-01` (resolved `2020-03-01`): `weak` low-rank evidence after row/column centering; top component energy `0.0542`, `95%` rank `97`.
- Support from `complete_cov_ge_30`, start `2020-06-07` (resolved `2020-06-07`): `weak` low-rank evidence after row/column centering; top component energy `0.0588`, `95%` rank `89`.
- Support from `complete_cov_ge_30`, start `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.0519`, `95%` rank `78`.
- Support from `complete_cov_ge_40`, start `2020-03-01` (resolved `2020-03-01`): `weak` low-rank evidence after row/column centering; top component energy `0.0542`, `95%` rank `97`.
- Support from `complete_cov_ge_40`, start `2020-06-07` (resolved `2020-06-07`): `weak` low-rank evidence after row/column centering; top component energy `0.0588`, `95%` rank `89`.
- Support from `complete_cov_ge_40`, start `2020-09-06` (resolved `2020-09-06`): `weak` low-rank evidence after row/column centering; top component energy `0.0519`, `95%` rank `78`.
