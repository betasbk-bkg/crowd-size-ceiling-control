#!/usr/bin/env python3
"""
localization_index_ci.py — phase-averaged error localisation index with bootstrap 95% CI.

Replaces the retracted corner peak-to-mean statistic. Index = RMSE in the segment-phase
windows 0–10% and 90–100% (around a corner/reversal) divided by RMSE in the 40–60% window
(mid-segment). Phase = arc-length position within one segment (zigzag tooth, square side,
circle quarter, lemniscate quarter). Reproduces the archived run (MC = 20,
B = 2000) that was never archived locally.

READS : engine only
WRITES: ../data/localization_index_ci.json
RUN   : cd code ; python localization_index_ci.py     (240 runs, ~1 min)
Seed  : SeedSequence([777, TAG_ID[tag], N, int(tr*1000), mc]) with a fixed name->id map
        (an earlier draft used abs(hash(tag)), which Python salts per process,
        so those values (zigzag 2.78, circle 1.00) are not reproducible; this archived run is).
"""
import json, numpy as np
import adversary_ladder as E
from zigzag_periodic import PeriodicZigzag

MC = 20; B = 2000; V = 5.0
TAG_ID = {'zigzag': 1, 'square': 2, 'circle': 3, 'lemniscate': 4}

def series(tj, N, tr, seed):
    rng = np.random.default_rng(seed)
    pos = tj.start(); vel = np.zeros(2); h = [pos.copy()]; pang = 0.
    cd = np.array([1., 0.]); er = np.empty(E.FRAMES); ar = np.empty(E.FRAMES)
    for f in range(E.FRAMES):
        if f % E.VOTE_INT == 0:
            di = max(0, len(h)-1-E.DELAY_F); dp = h[di]
            _, a0 = tj.closest(dp); lap = tj.at(a0+E.LOOK)
            d = lap-dp; n = np.linalg.norm(d)
            if n > 1e-10: d /= n
            ia = np.degrees(np.arctan2(d[1], d[0]))
            v = E.gen_votes_adv(ia, pang, tr, N, rng, 'T1', 0.0, None); pang = ia
            bl = E.DIRS[v].mean(axis=0); g = np.linalg.norm(bl); cd = bl/g if g > 1e-10 else np.array([1., 0.])
        vel += E.SMOOTH*(cd*V-vel); pos = pos+vel*E.DT; h.append(pos.copy())
        cp, a = tj.closest(pos); er[f] = np.linalg.norm(pos-cp); ar[f] = a
    return er, ar

def li_boot(tj, unit, N, tr, tag):
    ed = []; md = []
    for s in range(MC):
        er, ar = series(tj, N, tr, seed=int(np.random.SeedSequence([777, TAG_ID[tag], N, int(tr*1000), s]).generate_state(1)[0]))
        ph = (ar % unit)/unit; e2 = er**2
        m1 = (ph < 0.1) | (ph >= 0.9); m2 = (ph >= 0.4) & (ph < 0.6)
        ed.append(np.mean(e2[m1])); md.append(np.mean(e2[m2]))
    ed = np.array(ed); md = np.array(md)
    pt = np.sqrt(ed.mean()/md.mean())
    rng = np.random.default_rng(0)
    bs = [np.sqrt(ed[i].mean()/md[i].mean()) for i in rng.integers(0, MC, (B, MC))]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

cases = [('zigzag', PeriodicZigzag(), 7.0711), ('square', E.Square(), 20.0),
         ('circle', E.Circle(R=10), 2*np.pi*10/4), ('lemniscate', E.Lemniscate(), E.Lemniscate().circ/4)]
out = {}
print(f"{'traj':<12}{'unit(m)':>8}{'N':>5}{'tr':>6}{'index':>8}{'95% CI':>18}")
for nm, tj, unit in cases:
    for N, tr in [(200, 0.20), (5, 0.20), (200, 0.40)]:
        p, lo, hi = li_boot(tj, unit, N, tr, nm)
        out[f"{nm}_N{N}_tr{tr}"] = {'index': round(float(p), 3), 'ci': [round(float(lo), 3), round(float(hi), 3)]}
        print(f"  {nm:<10}{unit:8.2f}{N:5d}{tr:6.0%}{p:8.2f}   [{lo:.2f}, {hi:.2f}]", flush=True)
out['_config'] = {'MC': MC, 'B': B, 'windows': 'edge 0-10%+90-100% / mid 40-60%', 'model': 'T1 c=0 (uniform)'}
json.dump(out, open('../data/localization_index_ci.json', 'w'), indent=1)
print("-> ../data/localization_index_ci.json")
