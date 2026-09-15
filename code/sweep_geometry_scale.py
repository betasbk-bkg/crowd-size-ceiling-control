#!/usr/bin/env python3
"""
sweep_geometry_scale.py — geometry-parameter sweep.

QUESTION
--------
Are there scaling laws, or phase-transition-like behaviour, when the geometry
parameters R, h and a are varied? Is the crowd-size dependence systematically
related to the size and shape of the path?

The main campaign holds every geometry parameter fixed (R = 10, h = 10,
a = 7), so the four geometries are four points, not a sweep. This script varies
the scale parameter of each smooth geometry and asks how the crowd-size
dependence moves with it.

DESIGN
------
  circle      R in {5, 7.5, 10, 15, 20}
  square      h in {5, 7.5, 10, 15, 20}
  lemniscate  a in {3.5, 5, 7, 10.5, 14}
  x  N (12 levels)  x  tr {5, 20, 40} %  x  MC = 50  x  T0     = 27,000 runs

zigzag is excluded: its amplitude interacts with the open-path correction and
would confound the scale effect. Stated as a scope limit.

PREREGISTERED PREDICTION (from the quadrature model)
-----------------------------------------------------------------------------
  e^2 ~ e_lag^2 + e_adv^2,  e_lag = kappa_g v tau_eff,  e_adv = c_a tr v T_c / sqrt(N)

Geometry enters only through the lag floor, so:
  (P1) the ceiling asymptote falls with scale as a(x) = a0 + c/x
  (P2) the crowd coefficient b is independent of scale
  (P3) therefore N_c(delta; x) = (b / (delta a(x)))^2 grows with scale — larger,
       gently curved paths keep benefiting from bigger crowds; tight ones
       saturate early.

A circle pilot (MC = 5) gave a(R) = 0.608 + 1.193/R with R^2 = 0.965 and
b varying by CV = 0.03 across R. This run tests P1-P3 at MC = 50 on three
geometries.

VALIDITY RANGE
--------------
The agent travels v*tau = 2.17 m during the response delay and looks ahead
2.0 m, so tracking breaks down once the curvature scale approaches that
distance. A circle probe found failure at R = 1.5 (R/look-ahead = 0.8: tracking
error exceeded half the radius) and clean behaviour from R = 2.5 upward. This
script probes each geometry's lower boundary directly so the paper can state
the range in which the scaling claim holds rather than asserting one.

GATE G1
-------
  pass per geometry if  a(x) fit R^2 >= 0.90  and  CV of b across levels <= 0.10
Geometries that fail are excluded from the claim, not argued around. If all
three fail, R2's question is answered with an explicit scope limit instead.

Usage (from code/):
  python sweep_geometry_scale.py --probe     # validity boundaries only, ~2 min
  python sweep_geometry_scale.py --smoke     # pipeline check, ~2 min
  python sweep_geometry_scale.py             # full 27,000 runs
  python sweep_geometry_scale.py --geom circle
Output: data/sweep_geometry_scale.json
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

SPEED = 5.0
TAU_S = E.DELAY_F * E.DT
TAU_EFF = TAU_S - E.DT / np.log(1 - E.SMOOTH)
T_C = TAU_S + (E.VOTE_INT * E.DT) / 2
LOOK = E.LOOK

GEOM = {
    'circle':     ('R', [5.0, 7.5, 10.0, 15.0, 20.0], lambda x: E.Circle(R=x)),
    'square':     ('h', [5.0, 7.5, 10.0, 15.0, 20.0], lambda x: E.Square(h=x)),
    'lemniscate': ('a', [3.5, 5.0, 7.0, 10.5, 14.0], lambda x: E.Lemniscate(a=x)),
}
TRS = [0.05, 0.20, 0.40]
CAMPAIGN_SWEEP = 2027          # distinct campaign tag: never collides with released seeds


def seed(geom, level, N, tr, mc):
    """Own campaign tag so sweep cells cannot collide with the released grid."""
    return int(np.random.SeedSequence(
        [CAMPAIGN_SWEEP, C.TRAJ_ID[geom], int(round(level * 10)), N,
         int(round(tr * 1000)), mc]).generate_state(1)[0])


def cell(tobj, N, tr, mc_n, geom, level):
    rm = [E.sim(tobj, N, tr, seed=seed(geom, level, N, tr, i),
                model='T1', coherence=0.0, speed=SPEED)['rmse'] for i in range(mc_n)]
    a = np.array(rm)
    return {'rmse_mean': round(float(a.mean()), 4),
            'rmse_std': round(float(a.std()), 4),
            'rmse_ci95': round(float(1.96 * a.std() / np.sqrt(mc_n)), 4),
            'mc_runs': mc_n}


# ------------------------------------------------------------------ probe
def run_probe(mc_n=5):
    """Find where tracking breaks down, so the validity range is measured."""
    print("=== validity probe: v*tau = %.2f m travelled during the delay, "
          "look-ahead = %.1f m ===" % (SPEED * TAU_S, LOOK))
    Ns = [5, 20, 50, 200]
    X = np.column_stack([np.ones(len(Ns)), 1 / np.sqrt(Ns)])
    out = {}
    for g, (pname, levels, mk) in GEOM.items():
        probe = sorted(set([levels[0] * 0.3, levels[0] * 0.5] + levels + [levels[-1] * 2]))
        print(f"\n  {g} ({pname})")
        print(f"    {pname:>7}{'scale/look':>12}{'a':>9}{'b':>9}{'R2':>8}   verdict")
        rows = []
        for x in probe:
            tobj = mk(x)
            y = np.array([cell(tobj, N, 0.20, mc_n, g, x)['rmse_mean'] for N in Ns])
            a, b = np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - ((y - (a + b / np.sqrt(Ns))) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            fail = (a > 0.5 * x) or (r2 < 0.3 and abs(b) < 0.05)
            rows.append({'level': x, 'a': round(float(a), 4), 'b': round(float(b), 4),
                         'r2': round(float(r2), 3), 'tracking_failure': bool(fail)})
            print(f"    {x:7.2f}{x/LOOK:12.1f}{a:9.4f}{b:+9.3f}{r2:8.3f}   "
                  f"{'TRACKING FAILURE' if fail else 'ok'}")
        ok = [r['level'] for r in rows if not r['tracking_failure']]
        out[g] = {'param': pname, 'probe': rows,
                  'valid_from': min(ok) if ok else None,
                  'planned_levels': levels,
                  'planned_inside_valid_range': bool(ok and min(levels) >= min(ok))}
        print(f"    -> valid from {pname} = {out[g]['valid_from']}; "
              f"planned levels inside range: {out[g]['planned_inside_valid_range']}")
    return out


# ------------------------------------------------------------------ sweep
def fit_cells(res, g, levels, Ns):
    """Per (level, tr): ceiling fit a, b. Then a(x) = a0 + c/x across levels."""
    X = np.column_stack([np.ones(len(Ns)), 1 / np.sqrt(Ns)])
    per = {}
    for x in levels:
        for tr in TRS:
            y = np.array([res[f"{g}_x{x}_tr{tr:.2f}_N{N}"]['rmse_mean'] for N in Ns])
            a, b = np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - ((y - (a + b / np.sqrt(Ns))) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            per[(x, tr)] = (float(a), float(b), float(r2))
    return per


def analyse(res, g, levels, Ns):
    per = fit_cells(res, g, levels, Ns)
    inv = np.array([1.0 / x for x in levels])
    A = np.column_stack([np.ones(len(levels)), inv])
    print(f"\n  === {g} ===")
    print(f"    {'level':>7}" + "".join(f"{'a(tr%d%%)' % (t*100):>11}{'b':>8}" for t in TRS))
    for x in levels:
        row = "".join(f"{per[(x,t)][0]:11.4f}{per[(x,t)][1]:+8.3f}" for t in TRS)
        print(f"    {x:7.2f}" + row)

    summary = {}
    for tr in TRS:
        av = np.array([per[(x, tr)][0] for x in levels])
        bv = np.array([per[(x, tr)][1] for x in levels])
        c = np.linalg.lstsq(A, av, rcond=None)[0]
        pred = A @ c
        r2 = 1 - ((av - pred) ** 2).sum() / max(((av - av.mean()) ** 2).sum(), 1e-12)
        bcv = float(np.std(bv) / abs(np.mean(bv))) if abs(np.mean(bv)) > 1e-9 else float('inf')
        nc = {f"{int(d*100)}%": [round(float((bv[i] / (d * av[i])) ** 2), 1) for i in range(len(levels))]
              for d in (0.01, 0.02, 0.05)}
        summary[f"tr{tr:.2f}"] = {'a0': round(float(c[0]), 4), 'c': round(float(c[1]), 4),
                                  'a_fit_R2': round(float(r2), 3), 'b_mean': round(float(bv.mean()), 4),
                                  'b_CV': round(bcv, 3), 'N_c_by_delta': nc}
        print(f"    tr={tr:.0%}:  a(x) = {c[0]:.4f} + {c[1]:.4f}/x   R2={r2:.3f}   "
              f"b mean={bv.mean():+.3f} CV={bcv:.3f}")
        print(f"              N_c(1%) across levels: {nc['1%']}")

    r2s = [summary[k]['a_fit_R2'] for k in summary]
    cvs = [summary[k]['b_CV'] for k in summary]
    passed = bool(min(r2s) >= 0.90 and max(cvs) <= 0.10)
    print(f"    GATE G1: a-fit R2 min={min(r2s):.3f} (>=0.90), b CV max={max(cvs):.3f} (<=0.10) "
          f"-> {'PASS' if passed else 'FAIL'}")
    summary['gate_pass'] = passed
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--geom', choices=list(GEOM), default=None)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    print("=== Geometry-parameter sweep ===")
    print(f"  engine unmodified | campaign tag {CAMPAIGN_SWEEP} (no collision with released seeds)")
    print(f"  tau_eff={TAU_EFF:.4f}s  T_c={T_C:.4f}s  look-ahead={LOOK}m  v={SPEED}m/s")

    if a.probe:
        probe = run_probe()
        json.dump({'probe': probe}, open(os.path.join(DATA, 'sweep_validity_probe.json'), 'w'), indent=2)
        print(f"\n  saved {os.path.join(DATA,'sweep_validity_probe.json')}")
        return

    mc_n, Ns = (3, [5, 50, 200]) if a.smoke else (C.MC, C.NS)
    geoms = [a.geom] if a.geom else list(GEOM)
    if a.smoke:
        print("  SMOKE: pipeline check only, results not interpreted")

    res = {}
    t0 = time.time()
    n = 0
    for g in geoms:
        pname, levels, mk = GEOM[g]
        for x in levels:
            tobj = mk(x)
            for tr in TRS:
                for N in Ns:
                    res[f"{g}_x{x}_tr{tr:.2f}_N{N}"] = cell(tobj, N, tr, mc_n, g, x)
                    n += mc_n
            print(f"  {g} {pname}={x:<5} done  [{time.time()-t0:.0f}s, {n} runs]")

    print("\n=== analysis: does geometry scale move the crowd-size dependence? ===")
    summ = {g: analyse(res, g, GEOM[g][1], Ns) for g in geoms}
    passed = [g for g in geoms if summ[g]['gate_pass']]
    print(f"\n  G1 verdict: {len(passed)}/{len(geoms)} geometries pass -> "
          f"{'report ' + ', '.join(passed) if passed else 'no geometry passes; state scope limit'}")
    if passed and len(passed) < len(geoms):
        print(f"  excluded: {', '.join(g for g in geoms if g not in passed)} "
              f"(report only the geometries that pass, do not argue around failures)")

    path = os.path.join(DATA, 'sweep_geometry_scale.json')
    json.dump({'config': {'MC': mc_n, 'Ns': Ns, 'trolls': TRS, 'model': 'T0',
                          'geometries': {g: {'param': GEOM[g][0], 'levels': GEOM[g][1]} for g in geoms},
                          'excluded': 'zigzag (amplitude confounded with the open-path correction)',
                          'seed': f'SeedSequence([{CAMPAIGN_SWEEP}, traj_id, level*10, N, tr_milli, mc])',
                          'prediction': 'a(x)=a0+c/x ; b independent of x ; N_c(delta;x)=(b/(delta a(x)))^2',
                          'smoke': a.smoke},
               'results': res, 'summary': summ,
               'gate_g1': {'passed': passed, 'n_geom': len(geoms)},
               'metadata': {'total_runs': n, 'elapsed_sec': round(time.time() - t0, 1),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
              open(path, 'w'), indent=1)
    print(f"\n  {n} runs in {time.time()-t0:.0f}s -> {path}")


if __name__ == '__main__':
    main()
