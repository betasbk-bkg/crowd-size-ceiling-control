#!/usr/bin/env python3
"""
largeN_saturation.py — observed saturation crowd size.

QUESTION
--------
Is there a critical crowd size beyond which adding participants no longer
changes tracking error?

WHY THE MODEL ALONE CANNOT ANSWER IT
------------------------------------
The ceiling model RMSE(N) = a + b/sqrt(N) gives a closed form for a
tolerance-defined saturation size, N_c(delta) = (b/(delta*a))^2. Two problems:

  1. 1/sqrt(N) has no intrinsic critical point, so N_c exists only once a
     tolerance delta is declared. (This is also why the observed behaviour is a
     smooth crossover rather than a phase transition.)
  2. At delta = 1% the value exceeds the tested range (N <= 200) in 19 of the 34
     cells with b > 0, i.e. it is extrapolation.

A pilot at N = 400-3200 (5 conditions, MC = 30) showed the extrapolation is not
merely uncertain but biased: the fitted asymptote underestimates the measured
floor systematically -- circle tr = 40% fitted a = 0.7488 versus measured floor
0.8096 (-7.5%) -- and because N_c depends on (b/a)^2 the error is squared. For
circle tr = 40% the extrapolated N_c(1%) = 11,061 against an observed saturation
near N = 400: a 27-fold overestimate.

So the closed form stays as the analytical answer, and the number it produces is
replaced by direct measurement. That is what this script provides.

WHAT IT MEASURES
----------------
  N in {400, 800, 1600, 3200} x 4 geometries x 10 adversarial ratios x MC = 50
  = 8,000 runs, model T0

Reported per cell:
  observed floor      mean RMSE over N >= 800
  saturation onset    smallest tested N whose RMSE is within `tol` of the floor
  plateau spread      (max-min)/mean over N >= 400
  fitted-a bias       (floor - a_fit)/a_fit, a_fit from the N <= 200 campaign
  N_c ratio           extrapolated N_c(1%) / observed saturation onset

SCOPE — WHAT IS NOT DONE
------------------------
The published fits are NOT refitted over the extended range. The campaign design
is N in [5, 200] with MC = 50 at 12 levels; these new points use 4 levels and are
a separate validation. Mixing the two designs in one weighted fit shifts b
enough to flip its sign for zigzag (+0.063 -> -0.063), which is an artefact of
combining designs with different level spacing and Monte Carlo depth, not a
physical result. Table 4, Table 1, Fig. 1, the change-point analysis and the
ANOVA therefore remain exactly as published; this run adds one table.

One published quantity does move. The attainable reduction is defined against
the fitted asymptote, and the fitted asymptote is low, so the attainable figures
are slightly optimistic: circle tr = 40% reads 31.99% against 29.82% measured,
circle tr = 20% 17.07% against 14.41%. Elsewhere the gap is under 1 point. The
script prints both so the correction can be made against measurement rather than
against another fit.

Note on interpretation: over a 16-fold extension every pilot condition moved by
less than 2%, including a small *increase* for zigzag (-1.23% from N = 200 to
3200). That is within the band described as practical crowd-size
independence, and the sign is not meaningful at that magnitude; it is not
evidence of a benefit reversal.

Usage (from code/):
  python largeN_saturation.py --smoke      # 2 conditions, MC=5, ~2 min
  python largeN_saturation.py              # full 8,000 runs, ~50 min
  python largeN_saturation.py --tol 0.02
Output: data/largeN_saturation.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(os.path.dirname(HERE), 'data')

import adversary_ladder as E   # noqa: E402
import campaign as C           # noqa: E402
from zigzag_periodic import PeriodicZigzag   # noqa: E402

BIG = [400, 800, 1600, 3200]
SMALL = C.NS                       # 5..200, the published levels
TRS = C.TRS                        # 10 adversarial ratios
GEOMS = ['circle', 'square', 'lemniscate', 'zigzag']
SPEED = 5.0


def traj(name):
    """zigzag uses the corrected periodic path; others as released."""
    return {'circle': lambda: E.Circle(R=10),
            'square': lambda: E.Square(),
            'lemniscate': lambda: E.Lemniscate(),
            'zigzag': lambda: PeriodicZigzag()}[name]()


def published_fit(tj, tr, src):
    """Weighted a, b over the published N <= 200 levels only."""
    NSF = np.array(SMALL, float)
    X = np.column_stack([np.ones_like(NSF), 1 / np.sqrt(NSF)])
    y = np.array([src[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_mean'] for N in SMALL])
    sd = np.array([src[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_std'] for N in SMALL])
    W = np.diag(1 / (sd / np.sqrt(50)) ** 2)
    a, b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(a), float(b), float(y[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tol', type=float, default=0.01,
                    help='within this fraction of the floor counts as saturated')
    a_ = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    mc = 5 if a_.smoke else 50
    trs = [0.20, 0.40] if a_.smoke else TRS
    geoms = ['circle', 'zigzag'] if a_.smoke else GEOMS
    if a_.smoke:
        print("SMOKE: pipeline check only, results not interpreted")

    # published campaign: circle/square/lemniscate as released, zigzag corrected
    rel = json.load(open(os.path.join(DATA, 'campaign_main_mc50_fixed.json')))['results']
    fixp = os.path.join(DATA, 'campaign_main_mc50_fixed_zzfix.json')
    if not os.path.exists(fixp):
        raise SystemExit("run merge_and_reanalyze_zzfix.py first "
                         "(need the corrected zigzag campaign)")
    fix = json.load(open(fixp))['results']

    print("=== Experiment 3: observed saturation crowd size ===")
    print(f"  N = {BIG} x {len(geoms)} geometries x {len(trs)} ratios x MC = {mc}")
    print("  published fits are NOT refitted; this is a separate validation\n")

    res, rows = {}, []
    t0, n = time.time(), 0
    for tj in geoms:
        obj = traj(tj)
        src = fix if tj == 'zigzag' else rel
        for tr in trs:
            big = {}
            for N in BIG:
                rm = [E.sim(obj, N, tr, seed=C.make_seed(tj, N, tr, 'T0', i),
                            model='T1', coherence=0.0, speed=SPEED)['rmse']
                      for i in range(mc)]
                arr = np.array(rm)
                big[N] = {'rmse_mean': round(float(arr.mean()), 4),
                          'rmse_std': round(float(arr.std()), 4),
                          'rmse_ci95': round(float(1.96 * arr.std() / np.sqrt(mc)), 4),
                          'mc_runs': mc}
                res[f"{tj}_tr{tr:.2f}_N{N}"] = big[N]
                n += mc

            af, bf, r5 = published_fit(tj, tr, src)
            vals = np.array([big[N]['rmse_mean'] for N in BIG])
            floor = float(np.mean([big[N]['rmse_mean'] for N in BIG if N >= 800]))
            onset = next((N for N in BIG
                          if abs(big[N]['rmse_mean'] - floor) <= a_.tol * floor), None)
            spread = float((vals.max() - vals.min()) / vals.mean())
            nc_ext = (bf / (0.01 * af)) ** 2 if bf > 0 else None
            rows.append({
                'traj': tj, 'tr': tr, 'a_fit': round(af, 4), 'b_fit': round(bf, 4),
                'floor_obs': round(floor, 4),
                'a_bias_pct': round(100 * (floor - af) / af, 2),
                'saturation_onset': onset, 'plateau_spread_pct': round(100 * spread, 2),
                'attainable_fit_pct': round(100 * (bf / np.sqrt(5)) / (af + bf / np.sqrt(5)), 2),
                'attainable_obs_pct': round(100 * (r5 - floor) / r5, 2),
                'N_c_extrap_1pct': None if nc_ext is None else round(nc_ext, 0),
                'overestimate_factor': (None if (nc_ext is None or not onset)
                                        else round(nc_ext / onset, 1)),
            })
            print(f"  {tj:<11} tr={tr:5.1%}  floor={floor:.4f}  onset=N{onset}  "
                  f"spread={100*spread:4.2f}%  a-bias={100*(floor-af)/af:+5.2f}%  "
                  f"[{time.time()-t0:.0f}s]")

    print("\n=== observed saturation vs extrapolated N_c ===")
    print(f"{'traj':<11}{'tr':>6}{'onset':>7}{'floor':>9}{'N_c(1%)':>10}{'over':>8}"
          f"{'attain fit':>12}{'attain obs':>12}")
    for r in rows:
        nc = 'n/a' if r['N_c_extrap_1pct'] is None else f"{r['N_c_extrap_1pct']:.0f}"
        ov = '' if r['overestimate_factor'] is None else f"{r['overestimate_factor']:.1f}x"
        print(f"  {r['traj']:<9}{r['tr']:6.0%}{str(r['saturation_onset']):>7}"
              f"{r['floor_obs']:9.4f}{nc:>10}{ov:>8}"
              f"{r['attainable_fit_pct']:12.2f}{r['attainable_obs_pct']:12.2f}")

    ok = [r for r in rows if r['saturation_onset']]
    if ok:
        print(f"\n  saturation reached by N={max(r['saturation_onset'] for r in ok)} "
              f"in {len(ok)}/{len(rows)} cells (tolerance {a_.tol:.0%} of the floor)")
        sp = [r['plateau_spread_pct'] for r in rows]
        print(f"  plateau spread over N >= 400: {min(sp):.2f}% to {max(sp):.2f}%")
        ovs = [r['overestimate_factor'] for r in rows if r['overestimate_factor']]
        if ovs:
            print(f"  extrapolated N_c overestimates the observed onset by "
                  f"{min(ovs):.1f}x to {max(ovs):.1f}x (median {np.median(ovs):.1f}x)")
        bias = [r['a_bias_pct'] for r in rows]
        print(f"  fitted asymptote bias: {min(bias):+.2f}% to {max(bias):+.2f}% "
              f"(negative fit = floor above fit)")
        d = [r['attainable_fit_pct'] - r['attainable_obs_pct'] for r in rows]
        print(f"  attainable, fit minus observed: {min(d):+.2f} to {max(d):+.2f} points")

    path = os.path.join(DATA, 'largeN_saturation' + ('_smoke' if a_.smoke else '') + '.json')
    json.dump({'config': {'N_levels': BIG, 'MC': mc, 'geometries': geoms, 'trolls': trs,
                          'model': 'T0', 'tolerance': a_.tol,
                          'zigzag': 'PeriodicZigzag (corrected)',
                          'seed': 'campaign.make_seed — same convention as the published grid',
                          'refit': 'none; published N<=200 fits unchanged',
                          'smoke': a_.smoke},
               'results': res, 'summary': rows,
               'metadata': {'total_runs': n, 'elapsed_sec': round(time.time() - t0, 1),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
              open(path, 'w'), indent=1)
    print(f"\n  {n} runs in {time.time()-t0:.0f}s -> {path}")


if __name__ == '__main__':
    main()
