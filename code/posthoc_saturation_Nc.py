#!/usr/bin/env python3
"""
posthoc_saturation_Nc.py — where does the crowd-size benefit stop? (post-hoc)

STATUS: POST-HOC. This is not one of the preregistered predictions P1-P5 that
sweep_polygon_shape_size.py tests. Those were fixed before that run. The
question here came out of inspecting the first (R / h / a) sweep afterwards, and
the manuscript must label it as an exploratory observation, not a confirmed
prediction. Keeping it in a separate script keeps that distinction in the code.

THE OBSERVATION THAT PROMPTED IT
--------------------------------
In the first sweep the square failed the gate on the invariance of the
crowd coefficient b (CV 0.5-1.9). Reading the fit statistics alone, that
looks like noise. Reading the RMSE-vs-N curves instead shows something else:

    h = 7.5, 10     RMSE falls monotonically to N = 200   (ceiling R^2 0.94-0.97)
    h = 5, 15, 20   RMSE bottoms out at N ~ 10 and is flat after
                    (ceiling R^2 0.25-0.46, 5-6 non-decreasing steps out of 11)

So b was not unstable; the saturation point was jumping between N ~ 10 and
N > 200 as the size changed. The structural N-independence the manuscript
reports for the zigzag is not confined to one geometry: within the square it
switches on and off with the size parameter.

WHAT THIS SCRIPT MEASURES
-------------------------
Two saturation measures per cell, plus the diagnostics needed to tell a real
plateau from a poor fit:

  N_c(delta) = (b / (delta * a))^2      from the weighted ceiling fit; the crowd
                                        size beyond which the remaining gain is
                                        below delta of the asymptote
  N_min                                 crowd size at the observed RMSE minimum
  early saturation                      N_min <= 20 AND the spread over N >= 20
                                        is under `flat_tol` of the mean
  monotone steps                        how many of the 11 steps in N decrease
  b / SE                                whether b is resolvable at all

A cell where b is not resolvable and the curve is flat is a genuine plateau. A
cell where b is not resolvable but the curve still moves is noise, and the two
must not be reported the same way.

THEN IT ASKS WHETHER SATURATION IS SYSTEMATIC
---------------------------------------------
For the polygon sweep every cell has a dimensionless corner-resolution ratio

    Lambda = (P / n) / (v * tau)

If early saturation sorts by Lambda, the switch is geometric and can be stated
as a relationship. If it does not, the honest report is that the regime switch
is observed but what sets it is unresolved — and the script says so rather than
searching for a variable that happens to fit.

Runs on whichever sweep files are present; neither is required.

Usage (from code/ or analysis/):
  python posthoc_saturation_Nc.py
  python posthoc_saturation_Nc.py --flat-tol 0.02
Output: data/posthoc_saturation_report.json
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

V, TAU = 5.0, 26 / 60


def wfit(y, sd, Ns, mc):
    """Weighted ceiling fit a + b/sqrt(N); same weighting as table4_ceiling_params.py."""
    NSF = np.array(Ns, float)
    X = np.column_stack([np.ones_like(NSF), 1 / np.sqrt(NSF)])
    sem = np.asarray(sd) / np.sqrt(mc)
    W = np.diag(1 / sem ** 2)
    XtWX = X.T @ W @ X
    a, b = np.linalg.solve(XtWX, X.T @ W @ np.asarray(y))
    se = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    pred = a + b / np.sqrt(NSF)
    y = np.asarray(y)
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-15)
    return float(a), float(b), float(se[1]), float(r2)


def describe(y, Ns, flat_tol):
    y = np.asarray(y, float)
    i_min = int(np.argmin(y))
    tail = y[[j for j, N in enumerate(Ns) if N >= 20]]
    spread = float((tail.max() - tail.min()) / tail.mean())
    steps_down = int(sum(1 for i in range(len(y) - 1) if y[i + 1] < y[i] - 1e-12))
    return {'N_min': Ns[i_min], 'tail_spread': round(spread, 4),
            'steps_down': steps_down, 'n_steps': len(y) - 1,
            'early_saturation': bool(Ns[i_min] <= 20 and spread < flat_tol)}


def nc(a, b, d):
    return None if b <= 0 else round(float((b / (d * a)) ** 2), 1)


def analyse(cells, flat_tol):
    """cells: list of dicts with label, y, sd, Ns, mc, and optional lam."""
    out = []
    for c in cells:
        a, b, se, r2 = wfit(c['y'], c['sd'], c['Ns'], c['mc'])
        d = describe(c['y'], c['Ns'], flat_tol)
        out.append({**{k: c[k] for k in c if k not in ('y', 'sd', 'Ns', 'mc')},
                    'a': round(a, 4), 'b': round(b, 4), 'b_over_se': round(abs(b / se), 1),
                    'ceiling_R2': round(r2, 3),
                    'N_c_1pct': nc(a, b, 0.01), 'N_c_2pct': nc(a, b, 0.02),
                    'N_c_5pct': nc(a, b, 0.05), **d})
    return out


def report(rows, title, extra_cols=()):
    print(f"\n=== {title} ===")
    head = f"{'cell':<22}{'a':>8}{'b':>8}{'b/SE':>7}{'R2':>7}{'Nmin':>6}{'tail':>8}{'dn':>5}{'Nc1%':>9}  regime"
    print(head)
    for r in rows:
        reg = 'EARLY SATURATION' if r['early_saturation'] else (
            'gain to N=200' if r['b_over_se'] > 3 and r['N_min'] >= 75 else 'intermediate')
        ncv = f"{r['N_c_1pct']:9.0f}" if r['N_c_1pct'] is not None else f"{'n/a (b<=0)':>9}"
        print(f"  {r['label']:<20}{r['a']:8.4f}{r['b']:+8.3f}{r['b_over_se']:7.1f}"
              f"{r['ceiling_R2']:7.3f}{r['N_min']:6d}{r['tail_spread']:8.4f}"
              f"{r['steps_down']:5d}{ncv}  {reg}")


def lambda_sort(rows):
    """Does early saturation sort by Lambda? Report honestly either way."""
    have = [r for r in rows if r.get('lam') is not None and np.isfinite(r['lam'])]
    if not have:
        return None
    early = [r['lam'] for r in have if r['early_saturation']]
    late = [r['lam'] for r in have if not r['early_saturation']]
    print("\n=== does early saturation sort by Lambda = (P/n)/(v*tau)? ===")
    if not early or not late:
        print(f"  only one regime present ({len(early)} early / {len(late)} not) — "
              f"no separation to test")
        return {'separable': None, 'n_early': len(early), 'n_late': len(late)}
    lo_e, hi_e = min(early), max(early)
    lo_l, hi_l = min(late), max(late)
    clean = hi_e < lo_l or hi_l < lo_e
    print(f"  early saturation : Lambda in [{lo_e:.2f}, {hi_e:.2f}]  (n = {len(early)})")
    print(f"  continued gain   : Lambda in [{lo_l:.2f}, {hi_l:.2f}]  (n = {len(late)})")
    if clean:
        thr = (hi_e + lo_l) / 2 if hi_e < lo_l else (hi_l + lo_e) / 2
        print(f"  ranges do not overlap -> separable at Lambda ~ {thr:.2f}")
        print("  the regime switch is geometric and can be stated as a relationship.")
    else:
        print("  ranges overlap -> Lambda alone does not sort the two regimes.")
        print("  report the regime switch as observed, and state that what sets it")
        print("  is unresolved. Do not go looking for a variable that happens to fit.")
    return {'separable': bool(clean), 'early_range': [lo_e, hi_e], 'late_range': [lo_l, hi_l],
            'n_early': len(early), 'n_late': len(late)}


def load_polygon(path, flat_tol):
    d = json.load(open(path))
    if d['config'].get('smoke'):
        print(f"  skipping {os.path.basename(path)}: smoke run")
        return []
    R, Ns, mc = d['results'], d['config']['Ns'], d['config']['MC']
    cells = []
    for P in d['config']['perimeters']:
        for n in d['config']['n_sides']:
            lab = 'circle' if n == 0 else f"n{n}"
            for tr in d['config']['trolls']:
                pre = f"{lab}_P{int(P)}_tr{tr:.2f}"
                if f"{pre}_N{Ns[0]}" not in R:
                    continue
                cells.append({'label': f"{lab} P={int(P)} tr={int(tr*100)}%",
                              'lam': float('inf') if n == 0 else (P / n) / (V * TAU),
                              'y': [R[f"{pre}_N{N}"]['rmse_mean'] for N in Ns],
                              'sd': [R[f"{pre}_N{N}"]['rmse_std'] for N in Ns],
                              'Ns': Ns, 'mc': mc})
    return analyse(cells, flat_tol)


def load_size(path, flat_tol):
    d = json.load(open(path))
    if d['config'].get('smoke'):
        print(f"  skipping {os.path.basename(path)}: smoke run")
        return []
    R, Ns, mc = d['results'], d['config']['Ns'], d['config']['MC']
    cells = []
    for g, meta in d['config']['geometries'].items():
        for x in meta['levels']:
            for tr in d['config']['trolls']:
                pre = f"{g}_x{x}_tr{tr:.2f}"
                if f"{pre}_N{Ns[0]}" not in R:
                    continue
                cells.append({'label': f"{g} {meta['param']}={x} tr={int(tr*100)}%",
                              'lam': None,
                              'y': [R[f"{pre}_N{N}"]['rmse_mean'] for N in Ns],
                              'sd': [R[f"{pre}_N{N}"]['rmse_std'] for N in Ns],
                              'Ns': Ns, 'mc': mc})
    return analyse(cells, flat_tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flat-tol', type=float, default=0.02,
                    help='tail spread below which the curve counts as flat (default 0.02)')
    a = ap.parse_args()

    print("=== POST-HOC: saturation crowd size across the sweeps ===")
    print("  NOT a preregistered prediction. Label it as exploratory in the manuscript.")
    print(f"  early saturation := RMSE minimum at N <= 20 and tail spread < {a.flat_tol:.0%}")

    out = {'flat_tol': a.flat_tol}
    poly = os.path.join(DATA, 'sweep_polygon_shape_size.json')
    size = os.path.join(DATA, 'sweep_geometry_scale.json')

    rows_p = load_polygon(poly, a.flat_tol) if os.path.exists(poly) else []
    if rows_p:
        report(rows_p, 'polygon sweep (shape n x size P)')
        out['polygon'] = rows_p
        out['lambda_sort'] = lambda_sort(rows_p)
    else:
        print("\n  polygon sweep not found or smoke-only — run it for the Lambda test")

    rows_s = load_size(size, a.flat_tol) if os.path.exists(size) else []
    if rows_s:
        report(rows_s, 'size sweep (R / h / a) — the observation that prompted this')
        out['size'] = rows_s
        n_e = sum(1 for r in rows_s if r['early_saturation'])
        print(f"\n  early saturation in {n_e}/{len(rows_s)} cells")
        for g in ('circle', 'square', 'lemniscate'):
            sub = [r for r in rows_s if r['label'].startswith(g)]
            e = sum(1 for r in sub if r['early_saturation'])
            print(f"    {g:<12}{e}/{len(sub)}")

    if rows_p or rows_s:
        path = os.path.join(DATA, 'posthoc_saturation_report.json')
        json.dump(out, open(path, 'w'), indent=1)
        print(f"\n  saved {path}")


if __name__ == '__main__':
    main()
