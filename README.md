# When Crowd Size Stops Mattering — Simulation Code and Data

Reproducibility repository for:

> **When Crowd Size Stops Mattering: Characterizing the Ceiling Effect in Crowd-Sourced
> Continuous Control — A Simulation Study**
> BongKeun Song, Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, Germany
> Submitted to *Scientific Reports* (2026)

This repository contains the simulation engine, experiment scripts, raw result data,
and analysis code needed to reproduce every table and figure in the paper.

## What the study does

Using a crowd-sourced continuous-control simulator, the study characterizes **when**
the "wisdom of crowds" averaging benefit appears as a function of crowd size *N*.
The main finding is that crowd size is a **second-order, conditional** design factor:
the benefit of adding more participants depends on the adversarial ratio, the
trajectory geometry, and the adversary structure. On reversing geometries (zigzag)
the benefit largely vanishes, and a coordinated persistent adversary can neutralize
it even on smooth trajectories.

All results are simulation-based; no human-participant data are used.

## v2.0.0 changes (2026-07)

1. **Participant-composition fix.** The original honest-block allocation overflowed at
   (N = 5, tr = 30%), executing 6 participants instead of 5. `code/adversary_ladder.py`
   now carries a surgical overflow guard: the buggy cell is corrected while **every
   other cell remains bit-for-bit identical** to the original engine.
   The 12 affected cells (600 of 72,000 runs) were re-simulated
   (`analysis/bugfix_N5_tr30_rerun.py`) and the corrected campaign dataset is
   `data/campaign_main_mc50_fixed.json`. All paper statistics use the corrected data.
2. **Analysis scripts added** so that every reported statistic maps to an executable
   script (see mapping in `REPRODUCE.md`): ceiling fits (Table 1 / Supplementary
   Table S6), fitted ceiling parameters (Table 4), raw two-way ANOVA and assumption
   diagnostics (Table 2), change-point analysis, quadrature-model calibration, and
   the lag-axis sensitivity experiment (Table 3, `data/lagaxis_results.json`).
3. README/REPRODUCE updated from the earlier IEEE Access submission to the
   Scientific Reports version (display items renumbered).

## Repository layout

```
code/      simulation engine, campaign scripts
analysis/  paper-statistic scripts (v2.0.0) — every Table/Figure maps here
data/      raw Monte Carlo result data (JSON/CSV)
figures/   final figures (PDF + PNG)
```

### code/ (campaigns, unchanged from v1 except the engine guard)

| Script                      | Purpose                                              |
|-----------------------------|------------------------------------------------------|
| `adversary_ladder.py`       | Simulation engine + adversary models T0 / T1 / T2 (v2 overflow guard) |
| `campaign.py`               | Main 72,000-run campaign (dense grid, 3 adversaries) |
| `campaign_anova_raw.py`     | Raw-realization data for the two-way ANOVA           |
| `campaign_behavior.py`      | Behavioral-sensitivity panel                         |
| `campaign_speed_mech.py`    | Speed extension + mechanism slice                    |
| `campaign_mechanism.py`     | Time-resolved consensus / error traces               |
| `gate_10.py`                | Adversary-ladder verification gate                   |
| `geometry_descriptors.py`   | Trajectory geometric descriptors                     |

Initial-campaign (MC = 15) scripts `script1/2/3`, `sim_lemniscate_zigzag_troll15_3`,
`sim_v2_fullsweep` are retained unchanged.

### analysis/ (v2.0.0)

| Script                          | Reproduces                                        |
|---------------------------------|---------------------------------------------------|
| `ceiling_fit_tenlevel.py`       | Table 1 + Supplementary Table S6 (+ CSV)          |
| `table4_ceiling_params.py`      | Table 4 (a, b, 95% CIs, attainable reduction)     |
| `anova_diagnostics.py`          | Table 2 (η², F, p) + assumption diagnostics       |
| `changepoint_analysis.py`       | Transition-region change-points + bootstrap CIs   |
| `s5_quadrature_calibration.py`  | κ_g, c_a, R² and implied-b diagnostics            |
| `lagaxis_experiment.py`         | Table 3 (runs the lag-axis simulations, ~10 min)  |
| `bugfix_N5_tr30_rerun.py`       | The 600-run composition-fix re-simulation         |
| `make_fig1_heatmap.py`          | Figure 1 (R² heatmap, ten-level grid)             |

Seeding uses `numpy.random.SeedSequence` with fixed integer keys throughout;
every run is deterministic. See docstrings for the exact key layouts.

## Requirements

See `requirements.txt` (numpy, scipy, matplotlib).

## License / citation

MIT (see `LICENSE`). Cite via `CITATION.cff`.
