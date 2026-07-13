# Reproduction Guide (v2.0.0, Scientific Reports numbering)

Mapping from each paper element to the script and data that generate it.
Engine constants and the per-condition `SeedSequence` scheme make every run
bitwise reproducible. All paper statistics use the corrected campaign data
`data/campaign_main_mc50_fixed.json` (see README, v2.0.0 changes).

## Tables

| Paper element                  | Data file                          | Script                              |
|--------------------------------|------------------------------------|-------------------------------------|
| Table 1 (ceiling R², 6-level)  | `campaign_main_mc50_fixed.json`    | `analysis/ceiling_fit_tenlevel.py`  |
| Table 2 (ANOVA + diagnostics)  | `anova_raw_mc50.json`              | `analysis/anova_diagnostics.py`     |
| Table 3 (lag-axis sensitivity) | `lagaxis_results.json`             | `analysis/lagaxis_experiment.py`    |
| Table 4 (fitted a, b, CIs)     | `campaign_main_mc50_fixed.json`    | `analysis/table4_ceiling_params.py` |

## Text statistics

| Paper statement                         | Script                                |
|-----------------------------------------|---------------------------------------|
| Change-points 17% / 13% / 17% (+CIs)    | `analysis/changepoint_analysis.py`    |
| κ_g = 0.319, c_a = 1.67, R² = 0.889     | `analysis/s5_quadrature_calibration.py` |
| Zigzag ΔRMSE 0.07–1.96%, CV = 0.019     | direct from `campaign_main_mc50_fixed.json` (cell means) |

## Figures

| Figure                          | Data file                        | Script / notes                     |
|---------------------------------|----------------------------------|------------------------------------|
| Fig. 1 (R² heatmap, ten-level)  | `campaign_main_mc50_fixed.json`  | `analysis/make_fig1_heatmap.py`    |
| Fig. 2 (circle RMSE vs N)       | `campaign_main_mc50_fixed.json`  | three adversarial ratios           |
| Fig. 3 (zigzag N-independence)  | `campaign_main_mc50_fixed.json` + `mechanism_timeseries.json` | |
| Fig. 4 (adversary ladder)       | `campaign_main_mc50_fixed.json`  | tr = 40%, T0/T1/T2                 |
| Supp. Fig. S1 (ANOVA η²)        | `anova_raw_mc50.json`            |                                    |
| Supp. Fig. S2 (flowchart)       | —                                | schematic                          |

## Supplementary tables

| Element                     | Data / script                                        |
|-----------------------------|------------------------------------------------------|
| Supp. Table S2 (speed)      | `speed_ext.json`, `campaign_speed_mech.py`           |
| Supp. Table S3 (behavioral) | `behavior_panel.json`, `campaign_behavior.py`        |
| Supp. Table S4 (parameters) | values fixed in `adversary_ladder.py`                |
| Supp. Table S5 (campaigns)  | all `data/*.json` (accounting summary)               |
| Supp. Table S6 (ten-level R²)| `analysis/ceiling_fit_tenlevel.py` → `supplement_ceiling_fits_tenlevel.csv` |

## Initial (MC = 15) reproduction

Unchanged from v1: `script1/2/3`, `sim_lemniscate_zigzag_troll15_3`,
`sim_v2_fullsweep` produce the `E2f_*` / `E3_*` data files, each with a
sanity check against the reference data.
