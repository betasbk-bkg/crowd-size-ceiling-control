# When Crowd Size Stops Mattering — Simulation Code and Data

Reproducibility repository for:

> **When Crowd Size Stops Mattering: Characterizing the Ceiling Effect in Crowd-Sourced Continuous Control — A Simulation Study**
> BongKeun Song, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
> *IEEE Access* (Manuscript ID Access-2026-17026, under revision)

This repository contains the simulation engine, experiment scripts, raw result
data, and analysis code needed to reproduce all tables and figures in the paper.

## What the study does

Using a crowd-sourced continuous-control simulator, the study characterizes
**when** the "wisdom of crowds" averaging benefit appears as a function of
crowd size *N*. The main finding is that crowd size is a **second-order,
conditional** design factor: the benefit of adding more participants depends on
the adversarial ratio, the trajectory geometry, and the adversary structure.
On reversing geometries (zigzag) the benefit largely vanishes, and a
coordinated persistent adversary can neutralize it even on smooth trajectories.

All results are simulation-based; no human-participant data are used.

## Repository layout

```
code/      simulation engine, experiment scripts, analysis
data/      raw Monte Carlo result data (JSON)
figures/   final figures (PDF + PNG)   [optional, also in the paper]
```

### code/

Revised campaign (MC = 50; primary results):

| Script                      | Produces / Purpose                                   |
|-----------------------------|------------------------------------------------------|
| `adversary_ladder.py`       | Simulation engine + adversary models T0 / T1 / T2    |
| `campaign.py`               | Main 72,000-run campaign (dense grid, 3 adversaries) |
| `campaign_anova_raw.py`     | Raw-realization data for the two-way ANOVA (#5)      |
| `campaign_behavior.py`      | Behavioral-sensitivity panel (#9)                    |
| `campaign_speed_mech.py`    | Speed extension + mechanism slice (#13, #12)         |
| `campaign_mechanism.py`     | Time-resolved consensus / error traces (#12)         |
| `gate_10.py`                | Adversary-ladder verification gate (#10)             |
| `geometry_descriptors.py`   | Trajectory geometric descriptors (#11)               |

Initial campaign (MC = 15; reproduces the "initial analysis" referenced in the
revision):

| Script                               | Produces                          |
|--------------------------------------|-----------------------------------|
| `script1_sim_troll15_correct.py`     | `E3_troll15_correct.json`         |
| `script2_expA_e2f_mc15.py`           | `E2f_mc15.json`                   |
| `script3_expB_e3_raw_anova.py`       | `E3_raw_runs.json`, ANOVA         |
| `sim_lemniscate_zigzag_troll15_3.py` | `E2f_troll15.json`                |
| `sim_v2_fullsweep.py`                | `E3_v2_sweep.json`                |

### data/

Revised (MC = 50): `campaign_main_mc50.json`, `anova_raw_mc50.json`,
`behavior_panel.json`, `speed_ext.json`, `mech_mc50.json`,
`mechanism_timeseries.json`, `geometry_descriptors.json`.

Initial (MC = 15): `E2f_mc15.json`, `E2f_troll15.json`,
`E3_supplement_proper.json`, `E3_troll15_correct.json`, `E3_raw_runs.json`,
`E3_v2_sweep.json`.

## Engine constants

`DT = 1/60 s`, `FRAMES = 3900` (65 s), voting interval `18` frames (0.30 s),
exponential smoothing `alpha = 0.2`, response delay `26` frames (~0.433 s),
look-ahead `2.0`, nominal speed `5.0` units/s. Eight discrete voting
directions. Non-adversarial mix scaled as `sc = (1 - tr)/0.95`.

Each condition uses an independent NumPy `SeedSequence` stream keyed by
(trajectory, N, adversarial ratio, model, Monte Carlo index), ensuring
reproducible and mutually independent draws.

## Quick start

```bash
pip install -r requirements.txt

# main revised campaign (long-running; 72,000 runs)
python code/campaign.py

# raw-realization ANOVA data
python code/campaign_anova_raw.py

# geometric descriptors (fast)
python code/geometry_descriptors.py
```

See `REPRODUCE.md` for the table/figure → script → data mapping.

## License

Code: MIT. Data: CC BY 4.0. See `LICENSE`.

## Citation

See `CITATION.cff`.
