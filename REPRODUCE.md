# Reproduction Guide (v2.1.2)

Mapping from each element of the paper and supplement to the data file and script that produce
it. Engine constants and the per-condition `SeedSequence` scheme make every run deterministic.

All zigzag results use the corrected periodic path; all statistics use
`data/campaign_main_mc50_fixed_zzfix.json` unless noted. The pre-correction datasets are kept
under their original names for comparison.

Run scripts from inside their own directory (`cd code` or `cd analysis`); paths are relative.

## Main tables

| Element | Data | Script |
|---------|------|--------|
| Table 1 (ceiling R², six-level subset) | `campaign_main_mc50_fixed_zzfix.json` | `analysis/ceiling_fit_tenlevel.py` |
| Table 2 (two-way ANOVA, η², F, P) | `anova_raw_mc50.json` | `analysis/anova_diagnostics.py` |
| Table 3 (lag axis, τ and α) | `lagaxis_results_zzfix.json` | `analysis/lagaxis_experiment.py` (non-zigzag cells) + `code/rerun_zigzag_periodic.py --block 5` and `code/merge_and_reanalyze_zzfix.py` (corrected zigzag cells, merged) |
| Table 4 (fitted a, b, 95% CI, attainable) | `campaign_main_mc50_fixed_zzfix.json` | `analysis/table4_ceiling_params.py` |

## Main figures

| Element | Data | Script |
|---------|------|--------|
| Figure 1 (reference and realised paths) | engine, run live | `code/make_figures_revision.py --fig 1` |
| Figure 2a (b vs adversarial ratio) | `campaign_main_mc50_fixed_zzfix.json` | `code/make_figures_revision.py --fig 2` |
| Figure 2b (ceiling-fit R² heat map) | `campaign_main_mc50_fixed_zzfix.json` | same |
| Figure 3 (RMSE vs N to 3,200) | + `largeN_saturation.json` | `code/make_figures_revision.py --fig 3` |
| Figure 4 (adversary ladder) | `campaign_main_mc50_fixed_zzfix.json` | `code/make_figures_revision.py --fig 4` |

## Supplementary tables

| Element | Data | Script |
|---------|------|--------|
| S1 (companion comparison) | — | accounting summary; run counts from S5 |
| S2 (speed, and path rotation) | `speed_ext_zzfix.json`, `zigzag_lowspeed_probe.json` | `code/campaign_speed_mech.py`, `code/zigzag_lowspeed_probe.py` |
| S3 (behavioural panel, T0/T1/T2) | `behavior_panel_zzfix.json` | `code/campaign_behavior.py` |
| S4 (parameters) | — | values fixed in `code/adversary_ladder.py` |
| S5 (campaign summary) | all `data/*.json` | accounting summary |
| S6 (ceiling R², ten levels) | `campaign_main_mc50_fixed_zzfix.json` | `analysis/ceiling_fit_tenlevel.py` |
| S7 (zigzag before/after correction) | `zzfix_reanalysis_report.json` | `code/merge_and_reanalyze_zzfix.py` |
| S8 (saturation vs extrapolated N_c) | `largeN_saturation.json` | `code/largeN_saturation.py`, `code/posthoc_saturation_Nc.py` |
| S9 (polygon shape sweep) | `sweep_polygon_shape_size.json` | `code/sweep_polygon_shape_size.py` |
| S10 (lag axis at measured latency) | `lagaxis_measured_latency.json` | `code/lagaxis_measured_latency.py` |
| S11 (adversarial exponent) | `tr_exponent_fit.json` | `code/tr_exponent_fit.py` |

## Supplementary figures

| Element | Data | Script |
|---------|------|--------|
| S1 (ANOVA η²) | `anova_raw_mc50.json` | `code/make_figures_revision.py --fig S1` |
| S2 (pipeline) | — | `code/make_figures_revision.py --fig S2` (schematic) |
| S3 (floor vs radius) | `sweep_geometry_scale.json`, `circle_range_ext.json`, `vote_noise_floor.json` | `code/make_figures_revision.py --fig S3` |
| S4 (shape axis) | `sweep_polygon_shape_size.json` | `code/make_figures_revision.py --fig S4` |
| S5 (error localisation) | `localization_index_ci.json` | `code/make_figures_revision.py --fig S5` |

## Text statistics

| Statement | Source |
|-----------|--------|
| Change-points 17% / 13% / 17% with CIs | `analysis/changepoint_analysis.py` |
| κ_g = 0.319, c_a = 1.67, R² = 0.889 | `analysis/s5_quadrature_calibration.py` |
| b(tr) slopes 2.131 / 0.630 / 0.882 / 0.277, R² 0.85–0.94 | `analysis/table4_ceiling_params.py` |
| Zigzag ΔRMSE −1.57% to +3.45%, per-cell CV 0.012 | `zzfix_reanalysis_report.json` |
| Floor CV ≤ 0.0035 at N = 3,200 over tr = 5–40% | `largeN_saturation.json` |
| a(R) = a₀ + c/R, R² = 0.945 / 0.955 over 2.5–80 m | `code/circle_range_ext.py` |
| Vote-noise floor 0.663 / 0.673 m at R = 1,000 m | `code/vote_noise_floor.py` |
| σ_θ·√N = 10.6 / 23.4 / 43.4 at tr = 5 / 20 / 40% | `code/sigma_theta_sqrtN.py` |
| Localisation index 2.83 (zigzag) vs 1.00 (circle) | `code/localization_index_ci.py` |
| Corner peak 1.75 / 1.75 / 1.10 m at 120° / 90° / 45° | `code/corner_response.py` |
| Free tr exponent 0.458–0.683 | `code/tr_exponent_fit.py` |
| Low-speed zigzag sign reversal and its removal by 22.5° rotation | `code/zigzag_lowspeed_probe.py` |

## Run order for a full rebuild

```
cd code
python campaign.py                      # 72,000 runs (long)
python campaign_anova_raw.py
python campaign_behavior.py
python campaign_speed_mech.py
python campaign_mechanism.py
cd ../analysis
python lagaxis_experiment.py            # lag-axis cells, released zigzag path
cd ../code
python rerun_zigzag_periodic.py         # 23,400 runs — corrected zigzag cells for every campaign above
python merge_and_reanalyze_zzfix.py     # writes the *_zzfix.json files used by everything below
python zzfix_mechanism_descriptors.py
python sweep_geometry_scale.py
python circle_range_ext.py
python sweep_polygon_shape_size.py
python largeN_saturation.py
python posthoc_saturation_Nc.py
python lagaxis_measured_latency.py
python vote_noise_floor.py
python sigma_theta_sqrtN.py
python corner_response.py
python localization_index_ci.py
python tr_exponent_fit.py
python zigzag_lowspeed_probe.py         # or --block 0..3 then --merge
python make_figures_revision.py
cd ../analysis                          # all of these read the *_zzfix.json files
python ceiling_fit_tenlevel.py
python table4_ceiling_params.py
python anova_diagnostics.py
python changepoint_analysis.py
python s5_quadrature_calibration.py
python anova_raw_mc15.py
```

## Data files not tied to a single element

| File | Role |
|------|------|
| `campaign_main_mc50.json`, `campaign_main_mc50_fixed.json` | pre-correction campaign (v2.0.0), kept for comparison |
| `behavior_panel.json`, `speed_ext.json`, `mech_mc50.json`, `lagaxis_results.json`, `geometry_descriptors.json`, `mechanism_timeseries.json` | pre-correction counterparts of the `_zzfix` files |
| `bugfix_N5_tr30_results.json` | the 600-run participant-composition re-simulation (v2.0.0) |
| `mechanism_timeseries_zzfix.json` | time-resolved consensus and error traces on the corrected path |
| `sweep_validity_probe.json` | radius range over which the steering law tracks at all |
| `supplement_ceiling_fits_tenlevel.csv` | CSV export of Supplementary Table S6 |
| `E2f_mc15.json`, `E2f_troll15.json` | initial MC = 15 campaign, aggregation-method and adversary-ratio runs |
| `E3_raw_runs.json` | initial campaign raw per-run RMSE for the first ANOVA |
| `anova_raw_results.json` | MC = 15 ANOVA table (η² 23.3% circle, 9.4% square), from `analysis/anova_raw_mc15.py` |
| `E3_troll15_correct.json`, `E3_supplement_proper.json`, `E3_v2_sweep.json` | initial campaign supplement and speed sweep |

## Initial (MC = 15) reproduction

Unchanged from v1: `script1/2/3`, `sim_lemniscate_zigzag_troll15_3` and `sim_v2_fullsweep`
produce the `E2f_*` / `E3_*` data files, each with a sanity check against the reference data.
