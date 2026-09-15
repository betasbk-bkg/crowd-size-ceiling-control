#!/usr/bin/env python3
"""
vote_noise_floor.py — error floor in the curvature -> 0 limit (independent estimate of a0).

The size law a(R) = a0 + c/R has intercept a0. This measures the floor directly on a circle so
large that curvature is negligible (R = 1000 m; sagitta L^2/8R = 0.0005 m) at N = 3200, where the
crowd term is exhausted, for tr = 20% and 40%.  Compare with the fitted a0 in circle_range_ext.json.

READS : engine only     WRITES: ../data/vote_noise_floor.json
RUN   : cd code ; python vote_noise_floor.py     (2 x 50 runs at N = 3200; ~15 min)
Seed  : SeedSequence([2032, N, int(tr*1000), mc])
"""
import sys, json, time, numpy as np, adversary_ladder as E
SMOKE='--smoke' in sys.argv; MC=3 if SMOKE else 50; R=1000.0; N=3200; TRS=[0.20,0.40]
t0=time.time(); out={}
for tr in TRS:
    rm=[E.sim(E.Circle(R=R),N,tr,seed=int(np.random.SeedSequence([2032,N,int(tr*1000),i]).generate_state(1)[0]),model='T1',coherence=0.0,speed=5.0)['rmse'] for i in range(MC)]
    out[f"tr{tr:.2f}"]={'rmse_mean':round(float(np.mean(rm)),4),'rmse_std':round(float(np.std(rm)),4),'sem':round(float(np.std(rm)/np.sqrt(MC)),4),'MC':MC}
    print(f"  tr={tr:.0%}: floor = {np.mean(rm):.4f} +/- {np.std(rm):.4f} (sd), sem {np.std(rm)/np.sqrt(MC):.4f}  [{time.time()-t0:.0f}s]",flush=True)
json.dump({'config':{'R':R,'N':N,'trolls':TRS,'MC':MC,'model':'T1 c=0 (uniform)','sagitta_m':2.0**2/(8*R),'smoke':SMOKE},'results':out},open('../data/vote_noise_floor'+('_smoke' if SMOKE else '')+'.json','w'),indent=1)
print('-> ../data/vote_noise_floor.json')
