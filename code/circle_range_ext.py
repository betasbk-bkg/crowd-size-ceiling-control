#!/usr/bin/env python3
"""
circle_range_ext.py — circle radius extension (R = 2.5, 40, 80 m) for the size-scaling law.

WHY: sweep_geometry_scale.json covers R = 5..20 m (4-fold). To discriminate a0 + c/R from
a0 + c/R^2 the range is widened to 2.5..80 m (32-fold). This reproduces the run first made
in a container session on 2026-09-12 that was never archived locally.

READS : ../data/sweep_geometry_scale.json   (R = 5, 7.5, 10, 15, 20)
WRITES: ../data/circle_range_ext.json
RUN   : cd code ; python circle_range_ext.py          (3,600 runs; a few minutes)
        python circle_range_ext.py --smoke            (MC = 3, pipeline check)

Model: T1 with coherence 0.0 == uniform adversary (bit-identical to T0 in this engine).
Seed : SeedSequence([2029, int(R*10), N, int(tr*1000), mc]) — same as the container run.
"""
import sys, json, time, numpy as np
import adversary_ladder as E, campaign as C
from scipy.optimize import curve_fit

SMOKE = '--smoke' in sys.argv
MC = 3 if SMOKE else 50
NS = C.NS; NSF = np.array(NS, float); X = np.column_stack([np.ones_like(NSF), 1/np.sqrt(NSF)])
NEW = [2.5, 40.0, 80.0]; TRS = (0.20, 0.40)

def wls(y, sd, mc):
    sem = sd/np.sqrt(mc); W = np.diag(1/sem**2); return np.linalg.solve(X.T@W@X, X.T@W@y)

t0 = time.time(); res = {}
for R in NEW:
    for tr in TRS:
        y = []; sd = []
        for N in NS:
            rm = [E.sim(E.Circle(R=R), N, tr,
                        seed=int(np.random.SeedSequence([2029, int(R*10), N, int(tr*1000), i]).generate_state(1)[0]),
                        model='T1', coherence=0.0, speed=5.0)['rmse'] for i in range(MC)]
            y.append(float(np.mean(rm))); sd.append(float(np.std(rm)))
        res[f"{R}_{tr}"] = {'R': R, 'tr': tr, 'Ns': NS, 'rmse_mean': y, 'rmse_std': sd, 'MC': MC}
    print(f"  R={R} done [{time.time()-t0:.0f}s]", flush=True)

sw = json.load(open('../data/sweep_geometry_scale.json'))['results']
fits = {}
for tr in TRS:
    xs = []; ys = []
    for R in [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 40.0, 80.0]:
        if R in NEW:
            y = np.array(res[f"{R}_{tr}"]['rmse_mean']); sd = np.array(res[f"{R}_{tr}"]['rmse_std']); mc = MC
        else:
            y = np.array([sw[f"circle_x{R}_tr{tr:.2f}_N{N}"]['rmse_mean'] for N in NS])
            sd = np.array([sw[f"circle_x{R}_tr{tr:.2f}_N{N}"]['rmse_std'] for N in NS]); mc = 50
        a, b = wls(y, sd, mc); xs.append(R); ys.append(float(a))
    xs = np.array(xs); ys = np.array(ys)
    print(f"\n  tr={tr:.0%}  R: " + " ".join(f"{x:5.1f}" for x in xs))
    print(f"          a: " + " ".join(f"{v:5.3f}" for v in ys))
    f_tr = {'R': xs.tolist(), 'a': ys.tolist()}
    for nm, f, k in [('a0+c/R', lambda x: 1/x, 2), ('a0+c/R2', lambda x: 1/x**2, 2), ('a0+c/sqrtR', lambda x: 1/np.sqrt(x), 2)]:
        A = np.column_stack([np.ones(len(xs)), f(xs)]); cc = np.linalg.lstsq(A, ys, rcond=None)[0]
        pred = A@cc; r2 = 1-((ys-pred)**2).sum()/((ys-ys.mean())**2).sum()
        aic = len(xs)*np.log(max(((ys-pred)**2).mean(), 1e-12))+2*k
        f_tr[nm] = {'a0': float(cc[0]), 'c': float(cc[1]), 'R2': float(r2), 'AIC': float(aic)}
        print(f"    {nm:<11} R2={r2:6.3f} AIC={aic:7.1f}  a0={cc[0]:.3f} c={cc[1]:.3f}")
    try:
        p, _ = curve_fit(lambda x, a0, c, pw: a0+c/x**pw, xs, ys, p0=[0.6, 1.0, 1.0], maxfev=40000)
        pred = p[0]+p[1]/xs**p[2]; r2 = 1-((ys-pred)**2).sum()/((ys-ys.mean())**2).sum()
        aic = len(xs)*np.log(max(((ys-pred)**2).mean(), 1e-12))+2*3
        f_tr['a0+c/R^p'] = {'a0': float(p[0]), 'c': float(p[1]), 'p': float(p[2]), 'R2': float(r2), 'AIC': float(aic)}
        print(f"    a0+c/R^p    R2={r2:6.3f} AIC={aic:7.1f}  a0={p[0]:.3f} c={p[1]:.3f} p={p[2]:.2f}")
    except Exception as e:
        f_tr['a0+c/R^p'] = {'error': str(e)}
    fits[f"tr{tr:.2f}"] = f_tr

out = {'config': {'new_levels': NEW, 'trolls': list(TRS), 'Ns': NS, 'MC': MC, 'model': 'T1 c=0 (uniform)',
                  'seed': 'SeedSequence([2029, int(R*10), N, int(tr*1000), mc])', 'smoke': SMOKE},
       'results': res, 'fits_full_range': fits,
       'metadata': {'total_runs': len(NEW)*len(TRS)*len(NS)*MC, 'elapsed_sec': round(time.time()-t0, 1),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}}
path = '../data/circle_range_ext' + ('_smoke' if SMOKE else '') + '.json'
json.dump(out, open(path, 'w'), indent=1)
print(f"\n  {out['metadata']['total_runs']} runs in {time.time()-t0:.0f}s -> {path}")
