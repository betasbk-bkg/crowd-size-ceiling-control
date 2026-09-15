#!/usr/bin/env python3
"""
lagaxis_measured_latency.py — does the conditional structure survive a realistic delay?

WHY
---
The manuscript states that tau = 26 frames (~433 ms) "represents realistic
end-to-end latency in streaming-based crowd-control platforms". Checking the two
cited sources shows it does not: the deployed crowd-control system reports 854.6
ms and its authors give that as a lower bound, and chat-driven platforms run
20-30 s stream-to-chat with multi-second vote tallies. The honest rewrite is to
call 433 ms a favourable operating point and say that measured values are higher.

That rewrite has a cost. Table 3 already shows the crowd-size benefit collapsing
along the lag axis: over tau_eff 0.291 -> 0.592 s (a 2.03-fold change) the
lemniscate benefit falls from 27.3% to 1.3%. A measured 854.6 ms corresponds to
tau_eff ~ 0.93 s, which is 1.57x the largest value tested. A reader can do that
extrapolation unaided and ask whether the smooth-trajectory benefit survives at
all under realistic latency. Leaving the question open is worse than answering
it, and answering it costs ten minutes.

WHAT IT MEASURES
----------------
  tau = 52 frames (866.7 ms, close to the 854.6 ms reported measurement)
  4 geometries x tr in {20%, 40%} x N in {5, 200} x MC = 50 = 800 runs

Reported per cell: RMSE at N = 5 and N = 200, the percentage benefit, a
two-sided Welch t-test on the per-run samples, and the same quantities at the
design point (tau = 26) for comparison.

READING THE RESULT
------------------
Either outcome is usable, which is why it is worth running.

  Benefit survives  -> the lag reframing costs nothing, and the lag-domination
                       account extends to measured latency rather than stopping
                       at the design point.
  Benefit collapses -> state it as a scope limit: the crowd-size benefit
                       reported here requires latency well below what deployed
                       platforms currently achieve. That is a stronger and more
                       useful claim than silence.

The engine is untouched; only the delay constant changes, exactly as in the
Table 3 lag-axis experiment. Seeds use a distinct campaign tag so nothing can
collide with the released grid.

Usage (from code/):
  python lagaxis_measured_latency.py --smoke     # 2 min
  python lagaxis_measured_latency.py             # 800 runs, ~10 min
Output: data/lagaxis_measured_latency.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(os.path.dirname(HERE), 'data')

import adversary_ladder as E                 # noqa: E402
from zigzag_periodic import PeriodicZigzag   # noqa: E402

SPEED = 5.0
DT = E.DT
ALPHA = E.SMOOTH
VOTE_INT = E.VOTE_INT
TAU_DESIGN = 26
TAU_MEASURED = 52          # 866.7 ms, against the 854.6 ms reported measurement
GEOMS = ['circle', 'square', 'lemniscate', 'zigzag']
TRS = [0.20, 0.40]
NS = [5, 200]
CAMPAIGN_LAG = 2030        # own tag; cannot collide with the released seeds


def traj(name):
    return {'circle': lambda: E.Circle(R=10),
            'square': lambda: E.Square(),
            'lemniscate': lambda: E.Lemniscate(),
            'zigzag': lambda: PeriodicZigzag()}[name]()


def tau_eff(delay_frames):
    """Total effective lag: transport delay plus the smoothing time constant."""
    return delay_frames * DT - DT / np.log(1 - ALPHA)


def seed(tj, N, tr, delay, mc):
    return int(np.random.SeedSequence(
        [CAMPAIGN_LAG, abs(hash(tj)) % 9999, N, int(round(tr * 1000)), delay, mc]
    ).generate_state(1)[0])


def run_cell(tj, N, tr, delay, mc):
    """Runs the released engine with DELAY_F temporarily set to `delay`."""
    obj = traj(tj)
    prev = E.DELAY_F
    E.DELAY_F = delay
    try:
        vals = [E.sim(obj, N, tr, seed=seed(tj, N, tr, delay, i), model='T0',
                      coherence=1.0, speed=SPEED)['rmse'] for i in range(mc)]
    finally:
        E.DELAY_F = prev
    a = np.array(vals)
    return a


def welch(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # two-sided, normal approximation (df here is ~90, so the error is under 1%)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p), float(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--delay', type=int, default=TAU_MEASURED)
    a_ = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    mc = 5 if a_.smoke else 50
    geoms = ['circle', 'zigzag'] if a_.smoke else GEOMS
    trs = [0.40] if a_.smoke else TRS

    td, tm = tau_eff(TAU_DESIGN), tau_eff(a_.delay)
    print("=== lag-axis extension to measured latency ===")
    print(f"  design   tau = {TAU_DESIGN} frames = {TAU_DESIGN*DT*1000:.0f} ms  ->  tau_eff = {td:.3f} s")
    print(f"  measured tau = {a_.delay} frames = {a_.delay*DT*1000:.0f} ms  ->  tau_eff = {tm:.3f} s"
          f"   ({tm/td:.2f}x the design point)")
    print(f"  reference: 854.6 ms reported as a lower bound in a deployed system")
    print(f"  {len(geoms)} geometries x {len(trs)} ratios x 2 crowd sizes x MC = {mc}"
          f" x 2 delays = {len(geoms)*len(trs)*2*mc*2} runs")
    if a_.smoke:
        print("  SMOKE: pipeline check only, results not interpreted")
    print()

    res, rows = {}, []
    t0, total = time.time(), 0
    for tj in geoms:
        for tr in trs:
            cell = {}
            for delay in (TAU_DESIGN, a_.delay):
                s5 = run_cell(tj, 5, tr, delay, mc)
                s200 = run_cell(tj, 200, tr, delay, mc)
                total += 2 * mc
                t, p, df = welch(s5, s200)
                d = 100 * (s5.mean() - s200.mean()) / s5.mean()
                cell[delay] = {'rmse5': round(float(s5.mean()), 4),
                               'rmse200': round(float(s200.mean()), 4),
                               'sd5': round(float(s5.std()), 4),
                               'sd200': round(float(s200.std()), 4),
                               'benefit_pct': round(float(d), 2),
                               'welch_t': round(t, 3), 'welch_p': p,
                               'tau_eff': round(tau_eff(delay), 4), 'mc_runs': mc}
                res[f"{tj}_tr{tr:.2f}_delay{delay}"] = cell[delay]
            b0 = cell[TAU_DESIGN]['benefit_pct']
            b1 = cell[a_.delay]['benefit_pct']
            rows.append({'traj': tj, 'tr': tr, 'benefit_design': b0, 'benefit_measured': b1,
                         'retained_frac': round(b1 / b0, 3) if b0 else None,
                         'p_measured': cell[a_.delay]['welch_p'],
                         'floor_design': cell[TAU_DESIGN]['rmse200'],
                         'floor_measured': cell[a_.delay]['rmse200']})
            print(f"  {tj:<11} tr={tr:5.0%}  benefit {b0:6.2f}% -> {b1:6.2f}%"
                  f"   (p = {cell[a_.delay]['welch_p']:.4f})"
                  f"   floor {cell[TAU_DESIGN]['rmse200']:.3f} -> {cell[a_.delay]['rmse200']:.3f} m"
                  f"   [{time.time()-t0:.0f}s]")

    print("\n=== reading ===")
    kept = [r for r in rows if r['benefit_measured'] > 3.5]
    lost = [r for r in rows if r['benefit_measured'] <= 3.5]
    print(f"  benefit above the 3.5% practical-independence band: {len(kept)}/{len(rows)} conditions")
    if kept:
        print("    " + ", ".join(f"{r['traj']} tr={r['tr']:.0%} ({r['benefit_measured']:.1f}%)"
                                 for r in kept))
    if lost:
        print(f"  within or below that band: "
              + ", ".join(f"{r['traj']} tr={r['tr']:.0%} ({r['benefit_measured']:.1f}%)"
                          for r in lost))
    fr = [r['retained_frac'] for r in rows if r['retained_frac'] is not None]
    if fr:
        print(f"  fraction of the design-point benefit retained: "
              f"{min(fr):.2f} to {max(fr):.2f} (median {np.median(fr):.2f})")
    print("\n  If the smooth-trajectory benefit survives, the latency rewrite costs nothing.")
    print("  If it does not, report it as a scope limit rather than leaving it to be inferred.")

    path = os.path.join(DATA, 'lagaxis_measured_latency'
                        + ('_smoke' if a_.smoke else '') + '.json')
    json.dump({'config': {'tau_design_frames': TAU_DESIGN, 'tau_measured_frames': a_.delay,
                          'tau_eff_design': round(td, 4), 'tau_eff_measured': round(tm, 4),
                          'alpha': ALPHA, 'vote_interval': VOTE_INT, 'speed': SPEED,
                          'geometries': geoms, 'trolls': trs, 'Ns': NS, 'MC': mc,
                          'model': 'T0', 'zigzag': 'PeriodicZigzag (corrected)',
                          'seed': f'SeedSequence([{CAMPAIGN_LAG}, traj, N, tr_milli, delay, mc])',
                          'smoke': a_.smoke},
               'results': res, 'summary': rows,
               'metadata': {'total_runs': total, 'elapsed_sec': round(time.time() - t0, 1),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
              open(path, 'w'), indent=1)
    print(f"\n  {total} runs in {time.time()-t0:.0f}s -> {path}")


if __name__ == '__main__':
    main()
