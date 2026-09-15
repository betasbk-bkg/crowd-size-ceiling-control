#!/usr/bin/env python3
"""
sweep_polygon_shape_size.py — Experiment 2b: separating geometry SIZE from SHAPE.

QUESTION
--------
"the possible existence of scaling laws or phase-transition-like behaviour when
the control parameters, such as R, h, a, etc., are varied. Is there a systematic
relationship between these parameters / the shape and size of the geometry and
the observed dependence on crowd size?"

WHY THE RELEASED DATA CANNOT ANSWER IT
--------------------------------------
The submitted campaign fixes every geometry parameter (R = 10, h = 10, a = 7,
amp = 5) and treats geometry as a 4-level categorical factor. A relationship in
a variable that never varies is not estimable. Worse, the four geometries differ
in shape AND size simultaneously, so the two effects are perfectly confounded:
fitting a = a0 + c*(2*pi/P) to those four points gives R^2 = 0.011.

WHY THE FIRST SWEEP (R / h / a) ONLY PARTLY ANSWERED IT
-------------------------------------------------------
It varied size within each geometry, which is right, but applied a curvature-
scaling prediction to two geometries that have no single curvature scale:
  * a square's curvature is concentrated at 4 corners of fixed 90 deg; changing
    h changes corner SPACING, not curvature. There is no 1/h to scale with.
  * a lemniscate's curvature varies along the path (measured ratio of max to min
    curvature is effectively unbounded), so one parameter cannot summarise it.
Only the circle has curvature fully defined by one number, and only the circle
passed: a(R) = a0 + c/R with R^2 = 0.93 (tr = 20%) and 0.90 (tr = 40%), and b
independent of R (CV = 0.055, 0.069). The other two failed because the
prediction did not apply to them, not because the data were noisy.

THE FIX: A FAMILY WHERE SHAPE AND SIZE ARE SEPARATE AND BOTH CONTINUOUS
-----------------------------------------------------------------------
For ANY closed convex curve the total turning is exactly 2*pi, independent of
size and shape. So turning demand per unit arc length is 2*pi/P, and for a
circle this equals the curvature 1/R exactly. That makes P the natural SIZE
variable for every geometry, and leaves SHAPE as the question of whether that
fixed 2*pi of turning is spread uniformly or concentrated at n corners.

  SIZE   perimeter P             mean turning demand 2*pi/P
  SHAPE  side count n            turn per corner 2*pi/n, corner spacing P/n
         (n -> infinity is the circle, so the circle is the limit of the family
          and the released square is the n = 4 member)

Dimensionless corner-resolution ratio, the collapse variable:

    Lambda = (P / n) / (v * tau)      corner spacing measured in delay-lengths

Lambda >> 1: corners arrive far apart relative to the response delay and are
handled independently. Lambda ~ 1: the delay smears successive corners together.

PREREGISTERED PREDICTIONS (fixed before the run)
------------------------------------------------
  P1 SIZE   at fixed n, a(P) = a0 + c*(2*pi/P)              fit R^2 >= 0.90
  P2 SHAPE  at fixed P, a decreases monotonically in n
  P3 LIMIT  a(n = 24) within 5% of the circle at the same P
  P4 CROWD  b independent of both P and n                   CV <= 0.15
  P5 COLLAPSE  a, rescaled by size, is a single function of Lambda

P4 is tested ONLY where b is statistically resolvable (|b/SE| > 3). In the first
sweep the gate applied a CV criterion at tr = 5%, where b is near zero by design
(round(N*tr) = 0 for small N, the discretisation effect the manuscript already
documents for the lemniscate). A coefficient of variation on a near-zero
quantity is meaningless; that was a gate-design error, not a result. tr = 5% is
therefore excluded here.

Horizon commensurability was checked before this design: over the 65 s horizon
the number of laps ranges from 2.0 to 10.4 across the levels, but dropping the
first 10 s changes RMSE by only 0.8-1.0%, so where the horizon cuts a lap does
not confound the comparison. All members are closed curves, so the open-path
problem that affected the zigzag cannot occur here.

DESIGN     P in {40, 80, 160} x n in {3, 4, 6, 8, 12, 24, circle}
           x 12 crowd sizes x tr in {20%, 40%} x MC = 50   = 25,200 runs

Usage (from code/):
  python sweep_polygon_shape_size.py --probe    # geometry check, no simulation
  python sweep_polygon_shape_size.py --smoke    # pipeline check, ~2 min
  python sweep_polygon_shape_size.py            # full run, ~2 h
Output: data/sweep_polygon_shape_size.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(os.path.dirname(HERE), 'data')

import adversary_ladder as E   # noqa: E402
import campaign as C           # noqa: E402

SPEED = 5.0
TAU = E.DELAY_F * E.DT
PERIMS = [40.0, 80.0, 160.0]
NSIDES = [3, 4, 6, 8, 12, 24, 0]          # 0 = circle (n -> infinity)
TRS = [0.20, 0.40]
CAMPAIGN_POLY = 2028                       # own tag: cannot collide with released seeds


class RegularPolygon:
    """Closed regular n-gon of given perimeter. n -> infinity gives the circle.

    Same interface as adversary_ladder.Square; vertices are placed on the
    circumcircle so the shape is regular and the perimeter is exact.
    """

    def __init__(self, n, perimeter):
        self.n = int(n)
        side = perimeter / self.n
        Rc = side / (2 * np.sin(np.pi / self.n))          # circumradius
        th = np.arange(self.n + 1) * 2 * np.pi / self.n
        self.c = np.column_stack([Rc * np.cos(th), Rc * np.sin(th)])
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(self.n)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.] + list(np.cumsum(self.lens)))
        self.turn_per_corner = 360.0 / self.n
        self.corner_spacing = side

    def closest(self, p):
        bd, bp, ba = 1e18, self.c[0], 0.
        for i, (a, b) in enumerate(self.segs):
            v = b - a
            l2 = v @ v
            if l2 < 1e-10:
                continue
            t = np.clip((p - a) @ v / l2, 0, 1)
            pt = a + t * v
            d = np.linalg.norm(p - pt)
            if d < bd:
                bd, bp, ba = d, pt, self.cum[i] + t * self.lens[i]
        return bp, ba

    def at(self, arc):
        arc = arc % self.circ
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


def make(n, P):
    """n = 0 means the circle of the same perimeter (the n -> infinity member)."""
    return E.Circle(R=P / (2 * np.pi)) if n == 0 else RegularPolygon(n, P)


def label(n):
    return 'circle' if n == 0 else f"n{n}"


def lam(n, P):
    """Corner spacing in delay-lengths. Circle: uniform turning, no corners."""
    return float('inf') if n == 0 else (P / n) / (SPEED * TAU)


def seed(n, P, N, tr, mc):
    return int(np.random.SeedSequence(
        [CAMPAIGN_POLY, n, int(P), N, int(round(tr * 1000)), mc]).generate_state(1)[0])


def cell(tobj, n, P, N, tr, mc_n):
    rm = [E.sim(tobj, N, tr, seed=seed(n, P, N, tr, i), model='T1',
                coherence=0.0, speed=SPEED)['rmse'] for i in range(mc_n)]
    a = np.array(rm)
    return {'rmse_mean': round(float(a.mean()), 4), 'rmse_std': round(float(a.std()), 4),
            'rmse_ci95': round(float(1.96 * a.std() / np.sqrt(mc_n)), 4), 'mc_runs': mc_n}


# ------------------------------------------------------------------ fits
def fit_ab(res, key_prefix, Ns):
    NSF = np.array(Ns, float)
    X = np.column_stack([np.ones_like(NSF), 1 / np.sqrt(NSF)])
    y = np.array([res[f"{key_prefix}_N{N}"]['rmse_mean'] for N in Ns])
    sd = np.array([res[f"{key_prefix}_N{N}"]['rmse_std'] for N in Ns])
    sem = sd / np.sqrt(res[f"{key_prefix}_N{Ns[0]}"]['mc_runs'])
    W = np.diag(1 / sem ** 2)
    XtWX = X.T @ W @ X
    a, b = np.linalg.solve(XtWX, X.T @ W @ y)
    se = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    r2 = 1 - ((y - (a + b / np.sqrt(NSF))) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-15)
    return float(a), float(b), float(se[1]), float(r2)


def run_probe():
    print("=== geometry probe: is the design well posed? (no simulation) ===")
    print(f"  delay travel v*tau = {SPEED*TAU:.2f} m | look-ahead = {E.LOOK:.1f} m\n")
    print(f"{'n':>7}{'P':>7}{'side':>8}{'turn/corner':>13}{'Lambda':>9}{'laps in 65s':>13}")
    for P in PERIMS:
        for n in NSIDES:
            t = make(n, P)
            circ = getattr(t, 'circ', P)
            if n == 0:
                print(f"{'circle':>7}{P:7.0f}{'-':>8}{'uniform':>13}{'inf':>9}"
                      f"{SPEED*E.DUR/circ:13.2f}")
            else:
                print(f"{n:>7}{P:7.0f}{t.corner_spacing:8.2f}{t.turn_per_corner:12.1f}deg"
                      f"{lam(n,P):9.2f}{SPEED*E.DUR/circ:13.2f}")
        print()
    print("  total turning is 2*pi for every member, so size = P and shape = n are separable.")
    print("  Lambda = corner spacing / delay travel; the circle is the Lambda -> inf limit.")
    bad = [(n, P) for P in PERIMS for n in NSIDES if n and lam(n, P) < 1.0]
    print(f"  cells with Lambda < 1 (corners unresolvable by construction): "
          f"{bad if bad else 'none'}")


def analyse(res, Ns):
    rep = {'per_cell': {}}
    print("\n=== fitted a, b per cell ===")
    for tr in TRS:
        print(f"\n  tr = {tr:.0%}")
        print(f"{'n':>8}" + "".join(f"{'P=%d a' % P:>11}{'b':>9}{'b/SE':>7}" for P in PERIMS))
        for n in NSIDES:
            row = ""
            for P in PERIMS:
                a, b, se, r2 = fit_ab(res, f"{label(n)}_P{int(P)}_tr{tr:.2f}", Ns)
                rep['per_cell'][f"{label(n)}_P{int(P)}_tr{tr:.2f}"] = {
                    'a': round(a, 4), 'b': round(b, 4), 'b_over_se': round(abs(b / se), 1),
                    'ceiling_R2': round(r2, 3), 'lambda': lam(n, P)}
                row += f"{a:11.4f}{b:9.3f}{abs(b/se):7.1f}"
            print(f"{label(n):>8}" + row)

    # P1 size scaling at fixed n
    print("\n=== P1 SIZE: a(P) = a0 + c*(2pi/P) at fixed n ===")
    p1 = {}
    for tr in TRS:
        for n in NSIDES:
            xs = np.array([2 * np.pi / P for P in PERIMS])
            ys = np.array([rep['per_cell'][f"{label(n)}_P{int(P)}_tr{tr:.2f}"]['a'] for P in PERIMS])
            A = np.column_stack([np.ones(len(xs)), xs])
            c = np.linalg.lstsq(A, ys, rcond=None)[0]
            r2 = 1 - ((ys - A @ c) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-15)
            p1[f"{label(n)}_tr{tr:.2f}"] = {'a0': round(float(c[0]), 4),
                                            'c': round(float(c[1]), 3), 'R2': round(float(r2), 3)}
            print(f"  {label(n):>8} tr={tr:.0%}:  a0={c[0]:+.4f}  c={c[1]:+.3f}  R2={r2:.3f}"
                  f"   {'PASS' if r2 >= 0.90 else 'fail'}")
    rep['P1_size'] = p1

    # P2 shape monotonicity at fixed P
    print("\n=== P2 SHAPE: a decreasing in n at fixed P ===")
    p2 = {}
    for tr in TRS:
        for P in PERIMS:
            av = [rep['per_cell'][f"{label(n)}_P{int(P)}_tr{tr:.2f}"]['a'] for n in NSIDES]
            mono = all(av[i] >= av[i + 1] - 1e-9 for i in range(len(av) - 1))
            p2[f"P{int(P)}_tr{tr:.2f}"] = {'a_by_n': [round(v, 4) for v in av], 'monotone': bool(mono)}
            print(f"  P={P:5.0f} tr={tr:.0%}: " + " ".join(f"{v:.3f}" for v in av) +
                  f"   {'PASS (monotone)' if mono else 'fail (not monotone)'}")
    rep['P2_shape'] = p2

    # P3 circle limit
    print("\n=== P3 LIMIT: n = 24 vs circle at the same P ===")
    p3 = {}
    for tr in TRS:
        for P in PERIMS:
            a24 = rep['per_cell'][f"n24_P{int(P)}_tr{tr:.2f}"]['a']
            ac = rep['per_cell'][f"circle_P{int(P)}_tr{tr:.2f}"]['a']
            d = 100 * (a24 / ac - 1)
            p3[f"P{int(P)}_tr{tr:.2f}"] = {'n24': a24, 'circle': ac, 'diff_pct': round(d, 2)}
            print(f"  P={P:5.0f} tr={tr:.0%}: n24 {a24:.4f}  circle {ac:.4f}  "
                  f"diff {d:+.2f}%   {'PASS' if abs(d) < 5 else 'fail'}")
    rep['P3_limit'] = p3

    # P4 crowd coefficient invariance, resolvable cells only
    print("\n=== P4 CROWD: b invariance (|b/SE| > 3 cells only) ===")
    p4 = {}
    for tr in TRS:
        bs = [v['b'] for k, v in rep['per_cell'].items()
              if k.endswith(f"tr{tr:.2f}") and v['b_over_se'] > 3]
        n_res = len(bs)
        n_tot = len(NSIDES) * len(PERIMS)
        cv = float(np.std(bs) / abs(np.mean(bs))) if bs else float('inf')
        p4[f"tr{tr:.2f}"] = {'resolvable': n_res, 'total': n_tot, 'b_mean': round(float(np.mean(bs)), 4),
                             'b_CV': round(cv, 3)}
        print(f"  tr={tr:.0%}: resolvable {n_res}/{n_tot}  b mean={np.mean(bs):+.4f}  "
              f"CV={cv:.3f}   {'PASS' if cv <= 0.15 else 'fail'}")
    rep['P4_crowd'] = p4

    # P5 collapse in Lambda
    print("\n=== P5 COLLAPSE: does a - a_circle(P) collapse onto one curve in Lambda? ===")
    p5 = {}
    for tr in TRS:
        L, D = [], []
        for P in PERIMS:
            ac = rep['per_cell'][f"circle_P{int(P)}_tr{tr:.2f}"]['a']
            for n in NSIDES:
                if n == 0:
                    continue
                a = rep['per_cell'][f"{label(n)}_P{int(P)}_tr{tr:.2f}"]['a']
                L.append(lam(n, P)); D.append(a / ac - 1)
        L, D = np.array(L), np.array(D)
        try:
            g = lambda x, A, k: A * np.exp(-k * x)
            popt, _ = optimize.curve_fit(g, L, D, p0=[0.5, 0.2], maxfev=20000)
            pred = g(L, *popt)
            r2 = 1 - ((D - pred) ** 2).sum() / max(((D - D.mean()) ** 2).sum(), 1e-15)
            p5[f"tr{tr:.2f}"] = {'A': round(float(popt[0]), 4), 'k': round(float(popt[1]), 4),
                                 'R2': round(float(r2), 3), 'n_points': len(L)}
            print(f"  tr={tr:.0%}: (a/a_circle - 1) = {popt[0]:.3f} exp(-{popt[1]:.3f} Lambda)"
                  f"   R2={r2:.3f} over {len(L)} points   {'PASS' if r2 >= 0.80 else 'fail'}")
        except Exception as ex:
            p5[f"tr{tr:.2f}"] = {'error': str(ex)}
            print(f"  tr={tr:.0%}: collapse fit failed ({ex})")
    rep['P5_collapse'] = p5

    print("\n=== GATE G1' verdict ===")
    v = {
        'P1_size': all(x['R2'] >= 0.90 for x in p1.values()),
        'P2_shape': all(x['monotone'] for x in p2.values()),
        'P3_limit': all(abs(x['diff_pct']) < 5 for x in p3.values()),
        'P4_crowd': all(x['b_CV'] <= 0.15 for x in p4.values()),
        'P5_collapse': all(x.get('R2', 0) >= 0.80 for x in p5.values()),
    }
    for k, ok in v.items():
        print(f"  {k:<14}{'PASS' if ok else 'FAIL'}")
    npass = sum(v.values())
    print(f"\n  {npass}/5 predictions hold.")
    if npass == 5:
        print("  -> R2-3 CLOSED: size and shape both map systematically onto the")
        print("     crowd-size dependence, and the released square is the n = 4 member.")
    else:
        print("  -> report the predictions that hold; state the others as scope limits.")
        print("     Do not argue around a failed prediction.")
    rep['gate'] = v
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    print("=== Polygon shape/size sweep ===")
    print(f"  engine unmodified | campaign tag {CAMPAIGN_POLY} | v={SPEED} tau={TAU:.4f}s")
    if a.probe:
        run_probe()
        return

    mc_n, Ns = (3, [5, 50, 200]) if a.smoke else (C.MC, C.NS)
    if a.smoke:
        print("  SMOKE: pipeline check only, results not interpreted")
    run_probe()

    res = {}
    t0 = time.time()
    total = 0
    for P in PERIMS:
        for n in NSIDES:
            tobj = make(n, P)
            for tr in TRS:
                for N in Ns:
                    res[f"{label(n)}_P{int(P)}_tr{tr:.2f}_N{N}"] = cell(tobj, n, P, N, tr, mc_n)
                    total += mc_n
            print(f"  {label(n):>8} P={P:5.0f} done  [{time.time()-t0:.0f}s, {total} runs]")

    rep = analyse(res, Ns)
    path = os.path.join(DATA, 'sweep_polygon_shape_size.json')
    json.dump({'config': {'MC': mc_n, 'Ns': Ns, 'perimeters': PERIMS, 'n_sides': NSIDES,
                          'trolls': TRS, 'model': 'T0',
                          'size_var': 'perimeter P (mean turning demand 2*pi/P)',
                          'shape_var': 'side count n (turn per corner 2*pi/n)',
                          'collapse_var': 'Lambda = (P/n)/(v*tau)',
                          'seed': f'SeedSequence([{CAMPAIGN_POLY}, n, P, N, tr_milli, mc])',
                          'note': 'tr=5% excluded: b is near zero by design there',
                          'smoke': a.smoke},
               'results': res, 'analysis': rep,
               'metadata': {'total_runs': total, 'elapsed_sec': round(time.time() - t0, 1),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
              open(path, 'w'), indent=1)
    print(f"\n  {total} runs in {time.time()-t0:.0f}s -> {path}")


if __name__ == '__main__':
    main()
