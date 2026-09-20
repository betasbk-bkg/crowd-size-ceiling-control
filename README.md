# Crowd-Size Ceiling Effect in Crowd-Sourced Continuous Control — Simulation Code and Data

Reproducibility repository for:

> **Crowd size benefit in collective continuous control is conditional on adversarial ratio and trajectory geometry**
> BongKeun Song, Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, Germany
> *Scientific Reports* (under review, 2026)

This repository contains the simulation engine, experiment scripts, raw result data, analysis
code and figure scripts needed to reproduce every table and figure in the paper and its
supplement.

## What the study does

Using a crowd-sourced continuous-control simulator, the study asks when adding participants
improves tracking. The benefit of a larger crowd is conditional: it appears on smooth
trajectories once the adversarial ratio passes a transition region near tr ≈ 10–20%, it is
absent on a rapidly reversing (zigzag) path at every tested ratio, and a coordinated persistent
adversary removes it even on smooth paths. Extending the grid to N = 3,200 shows the benefit
narrowing beyond N = 400 without a critical crowd size.

All results are simulation-based; no human-participant data are used.

## v2.1.1 changes (2026-09-20)

Analysis-path correction only; no data changed. `analysis/ceiling_fit_tenlevel.py`, `table4_ceiling_params.py`, `changepoint_analysis.py` and `s5_quadrature_calibration.py` now read `campaign_main_mc50_fixed_zzfix.json` (the corrected campaign) instead of the pre-correction file; `supplement_ceiling_fits_tenlevel.csv` is regenerated from it; the Table 3 provenance is stated (merge step); the full-rebuild order runs the lag-axis experiment before the zigzag re-run and merge; the superseded heat-map script is removed.

## v2.1.0 changes (2026-09-16)

1. **Periodic zigzag path.** The released zigzag was an open polyline of length 70.71 m while
   the agent covers up to 325 m in the 65 s horizon, so on reaching the far end the look-ahead
   target wrapped to the path start while the projection clamped to the final segment; the agent
   then circulated on the last teeth for roughly two thirds of the horizon. `code/zigzag_periodic.py`
   extends the tooth pattern periodically, leaving the local geometry (segment 7.07 m, 90° vertex,
   correction window 3.26 × the response delay) unchanged. All 468 zigzag cells were re-simulated
   (`code/rerun_zigzag_periodic.py`, 23,400 runs) and merged (`code/merge_and_reanalyze_zzfix.py`).
   Two independent implementations — extending the released class and generating the pattern
   analytically — give identical RMSE in every checked cell. **Every non-zigzag entry is unchanged.**
   Supplementary Table S7 lists the affected quantities before and after.
2. **New campaigns.** Geometry-scale sweep and circle radius extension (30,600 runs),
   regular-polygon shape sweep (25,200), large-crowd extension to N = 3,200 (8,000), lag axis at
   the measured end-to-end latency τ = 52 frames (1,600), vote-noise floor, corner response,
   error-localisation index (~1,000), and a low-speed zigzag orientation probe (3,600).
3. **Figure scripts.** `code/make_figures_revision.py` regenerates all four main figures and
   Supplementary Figures S1–S5 from the archived data.
4. Documentation and table/figure numbering aligned with the current manuscript.

Datasets from v2.0.0 are retained unchanged alongside the corrected ones, so the two can be
compared directly (`*_zzfix.json` are the corrected versions).

## Repository layout

```
code/      simulation engine, campaign scripts, figure scripts
analysis/  paper-statistic scripts — every table and figure maps here or to code/
data/      raw Monte Carlo result data (JSON/CSV)
figures/   final figures (PDF + PNG)
```

The superseded v2.0.0 figures are not carried forward here; they remain in the v2.0.0 Zenodo
record (10.5281/zenodo.21337367).

### data/ — naming

Files whose name ends in `_zzfix` are the versions produced after the periodic-zigzag
correction and are the ones used in the paper; the same file without the suffix is the
v2.0.0 version, retained for comparison. `E2f_*` and `E3_*` are the initial MC = 15 campaign.
`sweep_validity_probe.json` records the radius range over which the steering law tracks at all
(failure at R = 1.5 m, clean from R = 2.5 m) and bounds the sweep in Supplementary Fig. S3.

In the polygon sweep, the circle rows carry `lambda: Infinity` by construction: Λ is the corner
spacing divided by the delay travel distance, and a circle has no corners. Those six entries are
excluded from the Λ regression reported in Supplementary Fig. S4b. JSON `Infinity` is non-standard;
`json.load` in Python reads it, stricter parsers may need `allow_nan`.

### code/ — engine and campaigns

| Script | Purpose |
|--------|---------|
| `adversary_ladder.py` | Simulation engine; adversary models T0 / T1 / T2; participant-composition guard |
| `zigzag_periodic.py` | Periodic zigzag path (v2.1.0 correction) |
| `campaign.py` | Main 72,000-run campaign (dense tr grid × three adversary models) |
| `campaign_anova_raw.py` | Raw per-run realisations for the two-way ANOVA |
| `campaign_behavior.py` | Behavioural-sensitivity panel (five honest-participant variants) |
| `campaign_speed_mech.py` | Speed extension (v = 2.0 m/s) and mechanism slice |
| `campaign_mechanism.py` | Time-resolved consensus and error traces |
| `geometry_descriptors.py` | Curvature, reversal and corner counts, correction window |
| `gate_10.py` | Adversary-ladder verification gate |

### code/ — v2.1.0 campaigns and measurements

| Script | Produces | Runs |
|--------|----------|------|
| `rerun_zigzag_periodic.py` | `zigzag_periodic_rerun.json` | 23,400 |
| `merge_and_reanalyze_zzfix.py` | `campaign_main_mc50_fixed_zzfix.json`, `zzfix_reanalysis_report.json` | — |
| `zzfix_mechanism_descriptors.py` | `mech_mc50_zzfix.json`, `geometry_descriptors_zzfix.json` | — |
| `sweep_geometry_scale.py` | `sweep_geometry_scale.json` | 27,000 |
| `circle_range_ext.py` | `circle_range_ext.json` (R = 2.5, 40, 80 m) | 3,600 |
| `sweep_polygon_shape_size.py` | `sweep_polygon_shape_size.json` | 25,200 |
| `largeN_saturation.py` | `largeN_saturation.json` (N = 400–3,200) | 8,000 |
| `posthoc_saturation_Nc.py` | `posthoc_saturation_report.json` | — |
| `lagaxis_measured_latency.py` | `lagaxis_measured_latency.json` (τ = 52) | 1,600 |
| `vote_noise_floor.py` | `vote_noise_floor.json` (R = 1,000 m, N = 3,200) | 100 |
| `sigma_theta_sqrtN.py` | `sigma_theta_sqrtN.json` (vote-level, no dynamics) | — |
| `corner_response.py` | `corner_response.json` (n-gon corner transient) | 60 |
| `localization_index_ci.py` | `localization_index_ci.json` (phase-resolved error) | 240 |
| `tr_exponent_fit.py` | `tr_exponent_fit.json` (analysis only) | — |
| `zigzag_lowspeed_probe.py` | `zigzag_lowspeed_probe.json` (path orientation vs vote grid) | 3,600 |
| `make_figures_revision.py` | Figures 1–4 and Supplementary Figures S1–S5 | — |

### analysis/ — paper statistics

| Script | Reproduces |
|--------|------------|
| `ceiling_fit_tenlevel.py` | Table 1 and Supplementary Table S6 |
| `table4_ceiling_params.py` | Table 4 (a, b, 95% CI, attainable reduction) |
| `anova_diagnostics.py` | Table 2 (η², F, P) and assumption diagnostics |
| `changepoint_analysis.py` | Transition-region change-points with bootstrap CIs |
| `s5_quadrature_calibration.py` | κ_g, c_a, R² and implied-b diagnostics |
| `lagaxis_experiment.py` | Lag-axis simulations (τ, α); the corrected Table 3 file `lagaxis_results_zzfix.json` is produced when `code/merge_and_reanalyze_zzfix.py` merges the re-run zigzag cells into its output |
| `bugfix_N5_tr30_rerun.py` | The 600-run participant-composition re-simulation |

Initial-campaign (MC = 15) scripts `script1/2/3`, `sim_lemniscate_zigzag_troll15_3` and
`sim_v2_fullsweep` produce the `E2f_*` / `E3_*` data files. Two of them opened a pre-v1 MC = 10 reference set (part1_e2f.json) that is not part of this archive, for an optional sanity check;
that check is now skipped when the file is absent and the campaign proceeds unchanged. The
analysis half of `script3_expB_e3_raw_anova.py` is also available on its own as
`analysis/anova_raw_mc15.py`, which regenerates `anova_raw_results.json` from the archived raw
runs without re-simulating.

## Reproducing

See `REPRODUCE.md` for the element-by-element mapping. Seeding uses `numpy.random.SeedSequence`
with fixed integer keys throughout, so every run is deterministic; the key layout for each
campaign is given in its script docstring.

Note on `localization_index_ci.py`: an earlier draft of this script seeded from Python's `hash()`
of the trajectory name, which is salted per process and therefore not reproducible. The archived
version uses a fixed name-to-integer map.

## Requirements

See `requirements.txt` (numpy, scipy, matplotlib).

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`).

## Archive

Archived on Zenodo. The concept DOI **10.5281/zenodo.20676802** covers all versions and always
resolves to the latest one; each release also has its own version DOI, shown on that release's
Zenodo record (v1.0.0: 10.5281/zenodo.20676803, v2.0.0: 10.5281/zenodo.21337367, v2.1.0: 10.5281/zenodo.22779940). The manuscript
cites the concept DOI together with the version label (v2.1.1). `CHECKSUMS.txt` lists SHA-256 for
every file in the release; it is generated over the archive contents and therefore does not list
itself.

## Citation

Cite via `CITATION.cff`.
