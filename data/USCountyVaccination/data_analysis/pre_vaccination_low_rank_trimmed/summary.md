# Pre-Vaccination Low-Rank Spectral Analysis

This analysis isolates the nationwide county-week death outcome panel before the start of vaccination and evaluates whether the observed outcome matrix looks empirically low rank under simple spectral diagnostics.

## Window

- Scope: `trimmed`
- Pre-vaccination cutoff: `WeekEndDate < 2020-12-27`
- First non-missing complete vaccination coverage date in processed panel: `2020-12-13`
- First strictly positive complete vaccination coverage date in processed panel: `2020-12-27`

## Matrices analyzed

- `Continuous death-rate matrix`: `48` weeks x `3009` counties from column `death_rate_100k`
- `Binary death-threshold matrix`: `48` weeks x `3009` counties from column `x_death_rate_100k_ge_2_pm1`

## Interpretation guide

- `Raw` includes level effects and common weekly shocks.
- `Row-centered` removes week-level means, so it downweights aggregate national waves.
- `Row-and-column centered` removes both week means and county means, so any remaining concentration is more suggestive of structured low-rank dependence rather than baseline levels.
- Low-rank labels here are descriptive: `strong`, `moderate`, and `weak` are based on spectral concentration, not on an exact-rank claim.

## Findings

### Continuous death-rate matrix

- Raw matrix: `weak` low-rank evidence; top component explains `0.3406` of spectral energy and `25` components explain `95%`.
- After row-and-column centering: `weak` low-rank evidence; top component explains `0.1992` of spectral energy and `27` components explain `95%`.
- Elbow summary after row-and-column centering: strongest adjacent singular-value drop at rank `1` with ratio `1.3506`.

### Binary death-threshold matrix

- Raw matrix: `weak` low-rank evidence; top component explains `0.4706` of spectral energy and `31` components explain `95%`.
- After row-and-column centering: `weak` low-rank evidence; top component explains `0.1268` of spectral energy and `34` components explain `95%`.
- Elbow summary after row-and-column centering: strongest adjacent singular-value drop at rank `2` with ratio `1.4087`.

## Outputs

- `continuous_death_spectrum.csv`
- `binary_death_spectrum.csv`
- `continuous_death_centering_comparison.csv`
- `binary_death_centering_comparison.csv`
- `continuous_death_scree.png`
- `continuous_death_cumulative_energy.png`
- `binary_death_scree.png`
- `binary_death_cumulative_energy.png`

