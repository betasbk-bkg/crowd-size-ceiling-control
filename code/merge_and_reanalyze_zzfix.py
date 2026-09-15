#!/usr/bin/env python3
"""
merge_and_reanalyze_zzfix.py — close the pipeline for the corrected zigzag.

WHY THIS EXISTS
---------------
rerun_zigzag_periodic.py writes its own file. Every analysis script in analysis/
hard-codes data/campaign_main_mc50_fixed.json (and the other campaign files) as
its input and takes no path argument. Without a merge step the re-run changes
nothing in Table 1, Table 4, Fig. 1, the change-point analysis or the quadrature
calibration: they would silently keep using the buggy zigzag cells.

The same pattern was used for the earlier (N = 5, tr = 30%) composition fix,
which produced data/bugfix_N5_tr30_results.json and was merged into
campaign_main_mc50_fixed.json. That merge was done by hand and left no script.
This one is scripted so the correction is reproducible.

WHAT IT DOES
------------
1. MERGE. Replaces only the zigzag entries in five campaign files, writing
   *_zzfix.json alongside the originals. Originals are never modified.
       campaign_main_mc50_fixed.json   360 zigzag cells (12 N x 10 tr x 3 models)
       speed_ext.json                   36 (v = 2.0)
       mech_mc50.json                    4
       behavior_panel.json              60
       lagaxis_results.json              4 (re-formed from the N=5/N=200 pair)

2. INTEGRITY GATE. Every non-zigzag entry must be bit-identical before and
   after. The script hashes them and aborts on any difference. This is what
   lets the paper say only the zigzag rows changed.

3. REANALYSIS on the merged data, each reported next to the released value:
       Table 1 / Table S6   ceiling-fit R^2, ten-level grid (unweighted OLS)
       Table 4              a, b, 95% CI, attainable (weighted LS, 1/sem^2)
       change-point         circle / square / lemniscate  (must be unchanged)
       quadrature           kappa_g, c_a per geometry  (released zigzag value
                            the released lag floor carried the terminal regime)
       N_c(delta)           (b / (delta a))^2 per geometry and tr
       per-cell CV          four-geometry ranking (manuscript: zigzag lowest)
       adversary ladder     T1 / T2 effect at tr = 40%, N = 200
       lag axis             delta% and Welch t/p from mean, sd, n
       Delta RMSE range     N = 5 -> 200 over the full tr grid

4. MANUSCRIPT DIFF. Prints every number the manuscript states about the zigzag
   with its corrected value, so the edit list is mechanical rather than manual.

Fits replicate the released scripts exactly:
  ceiling_fit_tenlevel.py   unweighted OLS on cell means
  table4_ceiling_params.py  weighted LS, weights 1/sem^2, sem = std/sqrt(50),
                            CIs from (X'WX)^-1 with z = 1.96
  changepoint_analysis.py   curve_fit per tr, two-segment breakpoint,
                            parametric bootstrap B = 400, seed 42

Usage (from code/ or analysis/):
  python merge_and_reanalyze_zzfix.py                 # merge + reanalyse
  python merge_and_reanalyze_zzfix.py --merge-only
  python merge_and_reanalyze_zzfix.py --no-bootstrap  # skip change-point CIs
Output: data/*_zzfix.json, data/zzfix_reanalysis_report.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

NS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
NSF = np.array(NS, float)
X = np.column_stack([np.ones_like(NSF), 1 / np.sqrt(NSF)])
TRS = [0.05, 0.10, 0.13, 0.15, 0.17, 0.20, 0.25, 0.30, 0.35, 0.40]
TRA = np.array(TRS)
MODELS = ['T0', 'T1', 'T2']
GEOMS = ['circle', 'square', 'lemniscate', 'zigzag']
MC = 50

# engine constants (shared with adversary_ladder.py)
DT, DELAY_F, SMOOTH, VOTE_INT, SPEED = 1 / 60, 26, 0.20, 18, 5.0
TAU = DELAY_F * DT
TAU_EFF = TAU - DT / np.log(1 - SMOOTH)
T_C = TAU + (VOTE_INT * DT) / 2


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def digest(results, skip_prefix='zigzag'):
    """Hash of every entry that must not change."""
    items = sorted((k, json.dumps(v, sort_keys=True))
                   for k, v in results.items() if not k.startswith(skip_prefix))
    return hashlib.sha256(''.join(k + v for k, v in items).encode()).hexdigest(), len(items)


# ------------------------------------------------------------------ 1. merge
def merge(rerun):
    R = rerun['results']
    reports = []

    def do(fname, mapping, label):
        src = load(fname)
        res = src['results'] if 'results' in src else src
        before, n_keep = digest(res)
        n_old = sum(1 for k in res if k.startswith('zigzag'))
        replaced, missing = 0, []
        for dst_key, val in mapping.items():
            if val is None:
                missing.append(dst_key)
                continue
            res[dst_key] = val
            replaced += 1
        after, n_keep2 = digest(res)
        if before != after or n_keep != n_keep2:
            raise SystemExit(f"INTEGRITY FAIL in {fname}: non-zigzag entries changed")
        out = fname.replace('.json', '_zzfix.json')
        with open(os.path.join(DATA, out), 'w') as f:
            json.dump(src, f, indent=1)
        status = 'OK' if not missing else f"MISSING {len(missing)}"
        print(f"  {label:<26} zigzag {n_old:>3} -> {replaced:>3} replaced | "
              f"non-zigzag {n_keep} unchanged (hash match) | {status} -> {out}")
        if missing:
            print(f"      missing keys (first 3): {missing[:3]}")
        reports.append({'file': fname, 'out': out, 'replaced': replaced,
                        'unchanged': n_keep, 'missing': missing})

    # main campaign
    m = {f"zigzag_tr{tr:.2f}_N{N}_{md}": R.get(f"zigzag_tr{tr:.2f}_N{N}_{md}")
         for tr in TRS for N in NS for md in MODELS}
    do('campaign_main_mc50_fixed.json', m, 'main campaign')

    # speed extension (v = 2.0)
    m = {f"zigzag_tr{tr:.2f}_N{N}_v2": R.get(f"zigzag_tr{tr:.2f}_N{N}_v2")
         for tr in [0.05, 0.20, 0.40] for N in NS}
    do('speed_ext.json', m, 'speed extension')

    # mechanism slice: runner key carries a _mech suffix
    m = {f"zigzag_N{N}_tr{tr:.2f}": R.get(f"zigzag_N{N}_tr{tr:.2f}_mech")
         for N in [5, 200] for tr in [0.05, 0.20]}
    do('mech_mc50.json', m, 'mechanism slice')

    # behaviour panel
    m = {f"zigzag_N{N}_tr{tr:.2f}_{md}_{b}": R.get(f"zigzag_N{N}_tr{tr:.2f}_{md}_{b}")
         for N in [30, 100] for tr in [0.15, 0.40]
         for md in MODELS for b in ['B0', 'B1', 'B2', 'B3', 'B4']}
    do('behavior_panel.json', m, 'behaviour panel')

    # lag axis: released format aggregates the N pair into one entry
    m = {}
    for a_, t_ in [(0.2, 13), (0.1, 13), (0.2, 26), (0.1, 26)]:
        lo, hi = R.get(f"zigzag_a{a_}_t{t_}_N5"), R.get(f"zigzag_a{a_}_t{t_}_N200")
        if lo and hi:
            d = (lo['rmse_mean'] - hi['rmse_mean']) / lo['rmse_mean'] * 100
            t = (lo['rmse_mean'] - hi['rmse_mean']) / np.sqrt(
                lo['rmse_std'] ** 2 / MC + hi['rmse_std'] ** 2 / MC)
            m[f"zigzag_a{a_}_t{t_}"] = {
                'rmse5': round(lo['rmse_mean'], 4), 'rmse200': round(hi['rmse_mean'], 4),
                'delta_pct': round(float(d), 2), 'welch_t': round(float(t), 3),
                'welch_p': float(2 * (1 - _norm_cdf(abs(t))))}
        else:
            m[f"zigzag_a{a_}_t{t_}"] = None
    do('lagaxis_results.json', m, 'lag axis')
    return reports


def _norm_cdf(z):
    import math
    return 0.5 * (1 + math.erf(float(z) / np.sqrt(2)))


# ------------------------------------------------------------- 2. reanalysis
def ols(y):
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - (a + b / np.sqrt(NSF))) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-15)
    return a, b, r2


def wls(y, sd):
    sem = sd / np.sqrt(MC)
    W = np.diag(1 / sem ** 2)
    XtWX = X.T @ W @ X
    a, b = np.linalg.solve(XtWX, X.T @ W @ y)
    se = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    att = 100 * (b / np.sqrt(5)) / (a + b / np.sqrt(5))
    return a, b, se, att


def cells(D, tj, tr, key='rmse_mean'):
    return np.array([D[f"{tj}_tr{tr:.2f}_N{N}_T0"][key] for N in NS])


def reanalyse(old, new, do_bootstrap=True):
    rep = {}
    print("\n=== Table 1 / S6: ceiling-fit R^2 (unweighted OLS), zigzag row ===")
    print(f"{'tr':>6}{'released':>11}{'corrected':>11}")
    row_o, row_n = [], []
    for tr in TRS:
        _, _, r2o = ols(cells(old, 'zigzag', tr))
        _, _, r2n = ols(cells(new, 'zigzag', tr))
        row_o.append(round(r2o, 3)); row_n.append(round(r2n, 3))
        print(f"{tr:6.0%}{r2o:11.3f}{r2n:11.3f}")
    rep['tableS6_zigzag_R2'] = {'released': row_o, 'corrected': row_n}

    print("\n=== Table 4: weighted fit (zigzag rows; circle shown for contrast) ===")
    t4 = {}
    for tj in ['circle', 'zigzag']:
        for tr in [0.05, 0.20, 0.40]:
            src = old if tj == 'circle' else new
            a, b, se, att = wls(cells(src, tj, tr), cells(src, tj, tr, 'rmse_std'))
            ao, bo, seo, atto = wls(cells(old, tj, tr), cells(old, tj, tr, 'rmse_std'))
            t4[f"{tj}_{tr}"] = {'a': round(a, 4), 'b': round(b, 3),
                                'b_ci': [round(b - 1.96 * se[1], 3), round(b + 1.96 * se[1], 3)],
                                'attainable_pct': round(att, 2),
                                'released': {'a': round(ao, 4), 'b': round(bo, 3),
                                             'attainable_pct': round(atto, 2)}}
            tag = '' if tj == 'circle' else '  <- corrected'
            print(f"  {tj:7s} tr={tr:.0%}: a={a:.4f}  b={b:+.3f} "
                  f"[{b-1.96*se[1]:+.3f},{b+1.96*se[1]:+.3f}]  attainable={att:6.2f}%"
                  f"   (released a={ao:.4f} b={bo:+.3f} {atto:.2f}%){tag}")
    rep['table4'] = t4

    print("\n=== Delta RMSE (N=5 -> 200), zigzag, full tr grid ===")
    dn = [(cells(new, 'zigzag', tr)[0] - cells(new, 'zigzag', tr)[-1]) /
          cells(new, 'zigzag', tr)[0] * 100 for tr in TRS]
    do_ = [(cells(old, 'zigzag', tr)[0] - cells(old, 'zigzag', tr)[-1]) /
           cells(old, 'zigzag', tr)[0] * 100 for tr in TRS]
    print(f"  released : {min(do_):+.2f}% .. {max(do_):+.2f}%   <- released version")
    print(f"  corrected: {min(dn):+.2f}% .. {max(dn):+.2f}%   (|max| = {max(abs(np.array(dn))):.2f}%)")
    rep['delta_rmse_range'] = {'released': [round(min(do_), 2), round(max(do_), 2)],
                               'corrected': [round(min(dn), 2), round(max(dn), 2)]}

    print("\n=== per-cell CV, four geometries (manuscript: zigzag lowest, 0.019) ===")
    cv = {}
    for tj in GEOMS:
        src = new if tj == 'zigzag' else old
        v = [src[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_std'] /
             src[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_mean'] for tr in TRS for N in NS]
        cv[tj] = round(float(np.mean(v)), 4)
        print(f"  {tj:<12}{cv[tj]:.4f}")
    lowest = min(cv, key=cv.get)
    print(f"  lowest = {lowest}  -> manuscript claim {'HOLDS' if lowest == 'zigzag' else 'FAILS'}")
    rep['per_cell_CV'] = cv

    print("\n=== quadrature calibration per geometry ===")
    print(f"{'geom':<12}{'kappa_g':>9}{'c_a':>7}{'R2':>7}{'lag floor':>11}")
    quad = {}
    for tj in GEOMS:
        src = new if tj == 'zigzag' else old
        Y, TR, NN = [], [], []
        for tr in TRS:
            for N in NS:
                Y.append(src[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_mean']); TR.append(tr); NN.append(N)
        Y, TR, NN = map(np.array, (Y, TR, NN))
        f = lambda p: np.sqrt((p[0] * SPEED * TAU_EFF) ** 2 +
                              (p[1] * TR * SPEED * T_C / np.sqrt(NN)) ** 2)
        r = optimize.least_squares(lambda p: f(p) - Y, [0.4, 1.5])
        kg, ca = r.x
        r2 = 1 - np.sum((Y - f(r.x)) ** 2) / np.sum((Y - Y.mean()) ** 2)
        quad[tj] = {'kappa_g': round(float(kg), 3), 'c_a': round(float(ca), 2),
                    'R2': round(float(r2), 3), 'lag_floor': round(float(kg * SPEED * TAU_EFF), 3)}
        print(f"  {tj:<12}{kg:9.3f}{ca:7.2f}{r2:7.3f}{kg*SPEED*TAU_EFF:11.3f}")
    rep['quadrature'] = quad

    print("\n=== N_c(delta) = (b / (delta a))^2, weighted fit ===")
    print(f"{'geom':<12}{'tr':>5}{'Nc 1%':>10}{'Nc 2%':>9}{'Nc 5%':>9}")
    nc = {}
    for tj in GEOMS:
        src = new if tj == 'zigzag' else old
        for tr in [0.05, 0.20, 0.40]:
            a, b, _, _ = wls(cells(src, tj, tr), cells(src, tj, tr, 'rmse_std'))
            vals = [(b / (d * a)) ** 2 if b > 0 else float('nan') for d in (0.01, 0.02, 0.05)]
            nc[f"{tj}_{tr}"] = [None if np.isnan(v) else round(float(v), 1) for v in vals]
            s = ''.join(f"{v:10.0f}" if not np.isnan(v) else f"{'n/a (b<0)':>10}" for v in vals)
            print(f"  {tj:<12}{tr:5.0%}{s}")
    rep['N_c'] = nc

    print("\n=== adversary ladder, zigzag at tr = 40%, N = 200 ===")
    lad = {}
    for md in MODELS:
        o = old[f"zigzag_tr0.40_N200_{md}"]['rmse_mean']
        n = new[f"zigzag_tr0.40_N200_{md}"]['rmse_mean']
        lad[md] = {'released': o, 'corrected': n}
        print(f"  {md}: released {o:.4f}  corrected {n:.4f}")
    eo = 100 * (old['zigzag_tr0.40_N200_T2']['rmse_mean'] /
                old['zigzag_tr0.40_N200_T0']['rmse_mean'] - 1)
    en = 100 * (new['zigzag_tr0.40_N200_T2']['rmse_mean'] /
                new['zigzag_tr0.40_N200_T0']['rmse_mean'] - 1)
    print(f"  T2 effect: released {eo:+.1f}%  corrected {en:+.1f}%"
          f"{'  (sign change)' if eo*en < 0 else ''}")
    rep['adversary_ladder'] = {'cells': lad, 'T2_effect_pct': {'released': round(eo, 1),
                                                               'corrected': round(en, 1)}}

    if do_bootstrap:
        print("\n=== change-point (must be unchanged: circle/square/lemniscate only) ===")
        cm = lambda N, a, b: a + b / np.sqrt(N)

        def bvec(D, tj, ym=None):
            out = []
            for j, tr in enumerate(TRS):
                y = ym[j] if ym is not None else cells(D, tj, tr)
                out.append(optimize.curve_fit(cm, NS, y, p0=[y.min(), 0.1])[0][1])
            return np.array(out)

        def cp(bs):
            best = None
            for bp in range(2, len(TRA) - 2):
                A1 = np.polyfit(TRA[:bp + 1], bs[:bp + 1], 1)
                A2 = np.polyfit(TRA[bp:], bs[bp:], 1)
                sse = (np.sum((np.polyval(A1, TRA[:bp + 1]) - bs[:bp + 1]) ** 2) +
                       np.sum((np.polyval(A2, TRA[bp:]) - bs[bp:]) ** 2))
                if best is None or sse < best[1]:
                    best = (TRA[bp], sse)
            return best[0]

        cps = {}
        for tj in ['circle', 'square', 'lemniscate']:
            po, pn = cp(bvec(old, tj)), cp(bvec(new, tj))
            cps[tj] = {'released': float(po), 'corrected': float(pn)}
            flag = 'unchanged' if po == pn else '*** CHANGED — investigate ***'
            print(f"  {tj:<12} released {po:.2f}  corrected {pn:.2f}   {flag}")
        rep['change_point'] = cps

    return rep


def manuscript_diff(rep):
    print("\n" + "=" * 72)
    print("ZIGZAG QUANTITIES — released -> corrected")
    print("=" * 72)
    d = rep['delta_rmse_range']
    print(f"  DeltaRMSE range  {d['released'][0]:+.2f}% .. {d['released'][1]:+.2f}%"
          f"  ->  {d['corrected'][0]:+.2f}% .. {d['corrected'][1]:+.2f}%")
    print(f"  per-cell CV      {rep['per_cell_CV'].get('zigzag_released', float('nan')):.4f}"
          f"  ->  {rep['per_cell_CV']['zigzag']:.4f}")
    for tr in [0.05, 0.20, 0.40]:
        t = rep['table4'][f"zigzag_{tr}"]
        r = t['released']
        print(f"  zigzag tr={tr:.0%}        a={r['a']:.4f} b={r['b']:+.3f} "
              f"{r['attainable_pct']:.2f}%  ->  a={t['a']:.4f} b={t['b']:+.3f} "
              f"{t['attainable_pct']:.2f}%")
    bs = [rep['table4'][f"zigzag_{tr}"]['b'] for tr in [0.05, 0.20, 0.40]]
    print(f"  zigzag b         {min(bs):+.3f} .. {max(bs):+.3f} (monotone in tr)")
    cb = rep['table4']['circle_0.4']['b']
    print(f"  b ratio circle/zigzag at tr=40%  ->  {cb / max(bs[-1], 1e-9):.1f}-fold")
    e = rep['adversary_ladder']['T2_effect_pct']
    print(f"  T2 effect on zigzag  {e['released']:+.1f}%  ->  {e['corrected']:+.1f}%")
    print(f"  ceiling R2, zigzag row       -> {rep['tableS6_zigzag_R2']['corrected']}")
    print(f"  quadrature kappa_g (zigzag)  {0.778:.3f} (released)  ->  "
          f"{rep['quadrature']['zigzag']['kappa_g']:.3f}")
    print("\n  Also regenerate: Fig 1 (zigzag row), Fig 3 (zigzag curve), Fig 4 (zigzag panel),")
    print("  SI Table S2 (zigzag rows), S3 (zigzag columns), S6 (zigzag row), Table 3 (zigzag column).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merge-only', action='store_true')
    ap.add_argument('--no-bootstrap', action='store_true')
    ap.add_argument('--rerun', default='zigzag_periodic_rerun.json')
    a = ap.parse_args()

    print("=== merge: corrected zigzag into the released campaign files ===")
    rerun = load(a.rerun)
    if rerun['config'].get('smoke'):
        raise SystemExit("refusing to merge a --smoke run")
    print(f"  source: {a.rerun}  ({rerun['metadata']['total_runs']} runs, "
          f"{rerun['config']['trajectory']})")
    xc = rerun.get('crosscheck_A_vs_F')
    if xc:
        print(f"  A/F cross-check on record: {xc['cells']} cells, "
              f"max|diff| = {xc['max_abs_diff']}")
    reports = merge(rerun)
    if a.merge_only:
        return

    old = load('campaign_main_mc50_fixed.json')['results']
    new = load('campaign_main_mc50_fixed_zzfix.json')['results']
    rep = reanalyse(old, new, do_bootstrap=not a.no_bootstrap)
    manuscript_diff(rep)

    out = os.path.join(DATA, 'zzfix_reanalysis_report.json')
    with open(out, 'w') as f:
        json.dump({'merge': reports, 'analysis': rep}, f, indent=1)
    print(f"\n  saved {out}")


if __name__ == '__main__':
    main()
