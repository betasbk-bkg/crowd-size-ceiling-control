#!/usr/bin/env python3
"""
zzfix_mechanism_descriptors.py — corrected-zigzag follow-ups (Experiments 1b, 1c).

The main re-run (rerun_zigzag_periodic.py) covers every campaign that produces
RMSE cells. Two reported quantities are produced by OTHER scripts and are not
covered by it:

  (1b) mechanism time series  -> code/campaign_mechanism.py
       Quantities affected: mean turn-rate proxy (~24 deg zigzag vs ~10 deg
       circle), corner error concentration, and the near-invariance of R-bar to
       crowd size. All are computed on the OPEN zigzag, so the terminal regime
       (67-68 % of the horizon) is baked into them.

  (1c) geometry descriptors   -> code/geometry_descriptors.py
       Manuscript reports "7 reversals per lap, 11 corners per lap,
       correction window ~3.3x the response delay". Two problems:
         * corner_count = ns + 1 = 11 counts the two endpoints of the OPEN
           polyline. A periodic path has no endpoints.
         * reversal_count is hard-coded to 4 in the script, while the manuscript
           says 7, and direct counting of y-direction sign changes gives 9.
           Three different numbers for the same quantity.
       This script recomputes all of them from the geometry and states the
       definition used, so the manuscript can be corrected against one source.

The engine is not modified. The released Zigzag is kept for side-by-side
reporting; PeriodicZigzag is the corrected path.

Outputs (data/):
  mechanism_timeseries_zzfix.json
  geometry_descriptors_zzfix.json

Usage (from code/):
  python zzfix_mechanism_descriptors.py            # both parts, ~3-6 min
  python zzfix_mechanism_descriptors.py --part 1b
  python zzfix_mechanism_descriptors.py --part 1c  # seconds, no simulation
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

import campaign as C                                   # noqa: E402
import adversary_ladder as E                           # noqa: E402
from campaign_mechanism import sim_trace               # noqa: E402
from zigzag_periodic import PeriodicZigzag             # noqa: E402

MC = 10
DELAY_S = E.DELAY_F * E.DT
VOTE_S = E.VOTE_INT * E.DT


# ---------------------------------------------------------------- 1b
def corner_concentration(err_series, turn_series, quantile=0.90):
    """Peak-to-mean error ratio at high-turn-demand frames.

    'Corner' frames are those in the top decile of instantaneous turn demand;
    the ratio is mean error there over mean error everywhere. This is the
    definition used for the manuscript's 2.6-vs-2.0 comparison.
    """
    e = np.asarray(err_series, float)
    t = np.asarray(turn_series, float)
    if np.all(t == 0):
        return float('nan')
    thr = np.quantile(t, quantile)
    m = t >= thr
    if m.sum() == 0:
        return float('nan')
    return float(e[m].mean() / e.mean())


def run_mechanism():
    """circle + zigzag(released) + zigzag(periodic) x N{5,200} x tr{5,20}%."""
    trajs = [('circle', E.Circle(), 'circle'),
             ('zigzag_open', E.Zigzag(), 'zigzag'),
             ('zigzag_periodic', PeriodicZigzag(), 'zigzag')]
    out = {}
    t0 = time.time()
    for label, tobj, seedname in trajs:
        for N in (5, 200):
            for tr in (0.05, 0.20):
                acc = {k: None for k in ['err_t', 'Rbar_t', 'turn_t', 'head_err_t']}
                summ = {k: [] for k in ['rmse', 'Rbar_mean', 'turn_mean', 'head_err_mean']}
                p2m = []
                for i in range(MC):
                    # seedname keeps the released seed stream: the corrected and
                    # released zigzag sit on the same random numbers.
                    seed = C.make_seed(seedname, N, tr, 'T0', i)
                    r = sim_trace(tobj, N, tr, seed)
                    for k in acc:
                        a = np.array(r[k])
                        acc[k] = a if acc[k] is None else acc[k] + a
                    for s in summ:
                        summ[s].append(r[s])
                    p2m.append(corner_concentration(r['err_t'], r['turn_t']))
                key = f"{label}_N{N}_tr{tr:.2f}"
                out[key] = {
                    **{f'{k}_mean_series': (acc[k] / MC).round(5).tolist() for k in acc},
                    **{s: round(float(np.mean(summ[s])), 4) for s in summ},
                    'corner_peak_to_mean': round(float(np.nanmean(p2m)), 3),
                    'MC': MC, 'N': N, 'tr': tr, 'traj': label,
                }
                print(f"  {key:<34} RMSE={out[key]['rmse']:.3f}  R̄={out[key]['Rbar_mean']:.3f}  "
                      f"turn={out[key]['turn_mean']:.1f}deg  p2m={out[key]['corner_peak_to_mean']:.2f}  "
                      f"[{time.time()-t0:.0f}s]")

    print("\n  --- manuscript quantities, released vs corrected ---")
    for N in (5, 200):
        for tr in (0.05, 0.20):
            c = out[f"circle_N{N}_tr{tr:.2f}"]
            o = out[f"zigzag_open_N{N}_tr{tr:.2f}"]
            p = out[f"zigzag_periodic_N{N}_tr{tr:.2f}"]
            print(f"  N={N:<4} tr={tr:.0%}  turn-rate  circle {c['turn_mean']:5.1f} | "
                  f"zz open {o['turn_mean']:5.1f} -> periodic {p['turn_mean']:5.1f} deg")
            print(f"  {'':<13}peak/mean  circle {c['corner_peak_to_mean']:5.2f} | "
                  f"zz open {o['corner_peak_to_mean']:5.2f} -> periodic {p['corner_peak_to_mean']:5.2f}")
    print("\n  --- R-bar invariance to N (manuscript claim) ---")
    for label in ('circle', 'zigzag_open', 'zigzag_periodic'):
        for tr in (0.05, 0.20):
            a = out[f"{label}_N5_tr{tr:.2f}"]['Rbar_mean']
            b = out[f"{label}_N200_tr{tr:.2f}"]['Rbar_mean']
            print(f"  {label:<16} tr={tr:.0%}  R̄(N=5)={a:.4f}  R̄(N=200)={b:.4f}  "
                  f"change {100*(b-a)/a:+.1f}%")

    path = os.path.join(DATA, 'mechanism_timeseries_zzfix.json')
    json.dump({'config': {'MC': MC,
                          'slice': 'circle / zigzag(open) / zigzag(periodic) x N{5,200} x tr{5,20}',
                          'model': 'T0',
                          'seed': 'campaign.make_seed with released trajectory name',
                          'corner_def': 'top-decile turn-demand frames; peak-to-mean error ratio',
                          'note': 'released and corrected zigzag on identical seed streams'},
               'results': out}, open(path, 'w'))
    print(f"\n  saved {path}  ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- 1c
def zigzag_geometry(amp=5.0, sx=5.0, ns=10, periodic=False):
    """Descriptors counted directly from the vertices, with stated definitions.

    reversal_count : sign changes of the y-component of the segment direction
                     per lap window (a 'reversal' is a change of vertical travel
                     direction).
    corner_count   : C1 discontinuities per lap window. An open polyline of ns
                     segments has ns-1 interior vertices; the released script
                     used ns+1, which counts the two endpoints as corners.
                     A periodic path has ns corners per lap window and no
                     endpoints.
    corr_window_s  : traversal time of one straight segment at nominal speed.
    """
    pts = [np.array([0.0, 0.0])]
    for i in range(ns):
        pts.append(np.array([(i + 1) * sx, amp if i % 2 == 0 else 0.0]))
    pts = np.array(pts)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    perim = float(seglen.sum())
    sign_changes = int(np.sum(np.diff(np.sign(seg[:, 1])) != 0))
    corners = ns if periodic else ns - 1
    corr = float(seglen.mean()) / E.MSPD
    return {
        'definition': 'periodic lap window' if periodic else 'open polyline',
        'mean_curvature': 0.0,
        'max_curvature': 'inf (vertex)',
        'reversal_count': sign_changes if not periodic else ns,
        'corner_count': corners,
        'segment_length': round(float(seglen.mean()), 4),
        'corr_window_s': round(corr, 4),
        'window_vs_delay': round(corr / DELAY_S, 2),
        'perimeter': round(perim, 2),
        'vertex_turn_deg': round(float(np.degrees(2 * np.arctan2(amp, sx))), 1),
    }


def run_descriptors():
    open_d = zigzag_geometry(periodic=False)
    peri_d = zigzag_geometry(periodic=True)
    print("=== zigzag descriptors ===")
    print(f"{'quantity':<20}{'open (released)':>18}{'periodic (fix)':>18}")
    for k in ['reversal_count', 'corner_count', 'segment_length',
              'corr_window_s', 'window_vs_delay', 'perimeter', 'vertex_turn_deg']:
        print(f"  {k:<18}{str(open_d[k]):>18}{str(peri_d[k]):>18}")
    print("\n  manuscript states: 7 reversals per lap, 11 corners per lap, ~3.3x window")
    print(f"  released script states: reversal_count = 4 (hard-coded), corner_count = ns+1 = 11")
    print("  -> three different reversal numbers exist (4 / 7 / counted). The counted")
    print("     value under the stated definition is authoritative; fix the manuscript")
    print("     and the released script to match, and state the definition explicitly.")
    print(f"\n  corner_count 11 counts the two endpoints of the OPEN path. With the")
    print(f"  corrected periodic path there are no endpoints: {peri_d['corner_count']} per lap window.")

    path = os.path.join(DATA, 'geometry_descriptors_zzfix.json')
    json.dump({'note': 'zigzag descriptors recomputed with explicit definitions; '
                       'open (released) vs periodic (corrected)',
               'delay_s': round(DELAY_S, 4), 'speed': E.MSPD,
               'zigzag_open': open_d, 'zigzag_periodic': peri_d}, open(path, 'w'), indent=2)
    print(f"\n  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['1b', '1c'], default=None)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.part in (None, '1c'):
        run_descriptors()
        print()
    if a.part in (None, '1b'):
        print("=== mechanism time series (MC=10, 12 conditions) ===")
        run_mechanism()


if __name__ == '__main__':
    main()
