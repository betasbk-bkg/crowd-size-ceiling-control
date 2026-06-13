# Reproduction Guide

Mapping from each paper element to the script and data that generate it.
Engine constants and the per-condition `SeedSequence` scheme make every run
bitwise reproducible.

## Tables

| Paper element        | Data file                          | Script                          |
|----------------------|------------------------------------|---------------------------------|
| Table I (parameters) | —                                  | values fixed in `adversary_ladder.py` |
| Table II (campaigns) | all `data/*.json`                  | accounting summary               |
| Table III (ceiling)  | `campaign_main_mc50.json`          | `campaign.py` + ceiling fit      |
| Table IV (ANOVA)     | `anova_raw_mc50.json`              | `campaign_anova_raw.py`          |
| Table V (speed)      | `speed_ext.json`, `campaign_main_mc50.json` | `campaign_speed_mech.py` |

## Figures

| Figure                       | Data file                  | Notes                         |
|------------------------------|----------------------------|-------------------------------|
| Fig. 1 (heatmap, R^2)        | `campaign_main_mc50.json`  | ceiling R^2 over the dense grid (MC = 50) |
| Fig. 2 (ANOVA eta^2)         | `anova_raw_mc50.json`      | raw-realization decomposition (MC = 50) |
| Fig. 3 (circle RMSE vs N)    | `campaign_main_mc50.json`  | three adversarial ratios       |
| Fig. 4 (zigzag N-independence)| `campaign_main_mc50.json` | + `mechanism_timeseries.json`  |
| Fig. S1 (trajectories)       | `geometry_descriptors.json`| target paths + descriptors     |

Figures are provided as PDF/PNG with the paper; the data above regenerate them.

## Reviewer-point → evidence

| Point | Addressed by                                            |
|-------|---------------------------------------------------------|
| #3 transition region   | `campaign.py` (dense tr grid) + change-point analysis |
| #5 raw ANOVA + robust  | `campaign_anova_raw.py` (`anova_raw_mc50.json`)       |
| #6 MC=50 stability     | `campaign.py` vs initial MC=15 data                   |
| #9 behavioral panel    | `campaign_behavior.py` (`behavior_panel.json`)        |
| #10 adversary ladder   | `adversary_ladder.py`, `gate_10.py`                   |
| #11 geometry descriptors | `geometry_descriptors.py` (`geometry_descriptors.json`) |
| #12 mechanism          | `campaign_mechanism.py` (`mechanism_timeseries.json`) |
| #13 speed extension    | `sim_v2_fullsweep.py`, `campaign_speed_mech.py`       |

## Initial (MC = 15) reproduction

The initial-submission analysis (referenced as "extended to MC = 50" in the
revision) is reproduced by the `script1/2/3`, `sim_lemniscate_zigzag_troll15_3`,
and `sim_v2_fullsweep` scripts, producing the `E2f_*` / `E3_*` data files.
Each script performs a sanity check against the reference data before running.
