#!/usr/bin/env python3
"""
corner_response.py — direct measurement of the error transient at a corner.

Tests the premises of a per-corner transient model: (i) peak error proportional to turn angle,
(ii) constant recovery time. Regular n-gons of equal side length (corner spacing = 20 m, so the
delay travel distance is the same fraction of every side): n = 3 (120 deg), 4 (90 deg), 8 (45 deg).
Error is phase-averaged along the side after each corner; peak = max of the phase-averaged
error, recovery = arc distance from the corner until the error first falls within 10% of the
mid-side level (phase 40-60%).

READS : engine only     WRITES: ../data/corner_response.json
RUN   : cd code ; python corner_response.py     (3 x 20 runs; ~2 min)
Seed  : SeedSequence([2033, n, N, int(tr*1000), mc])
"""
import json, numpy as np, adversary_ladder as E
from sweep_polygon_shape_size import RegularPolygon
SIDE=20.0; NSIDES=[3,4,8]; N=200; TR=0.20; MC=20; NB=50   # NB phase bins per side
def series(tj,seed):
    rng=np.random.default_rng(seed); pos=tj.start(); vel=np.zeros(2); h=[pos.copy()]; pang=0.; cd=np.array([1.,0.])
    er=np.empty(E.FRAMES); ar=np.empty(E.FRAMES)
    for f in range(E.FRAMES):
        if f%E.VOTE_INT==0:
            di=max(0,len(h)-1-E.DELAY_F); dp=h[di]; _,a0=tj.closest(dp); lap=tj.at(a0+E.LOOK)
            d=lap-dp; n=np.linalg.norm(d)
            if n>1e-10: d/=n
            ia=np.degrees(np.arctan2(d[1],d[0])); v=E.gen_votes_adv(ia,pang,TR,N,rng,'T1',0.0,None); pang=ia
            bl=E.DIRS[v].mean(axis=0); g=np.linalg.norm(bl); cd=bl/g if g>1e-10 else np.array([1.,0.])
        vel+=E.SMOOTH*(cd*5.0-vel); pos=pos+vel*E.DT; h.append(pos.copy())
        cp,a=tj.closest(pos); er[f]=np.linalg.norm(pos-cp); ar[f]=a
    return er,ar
out={}
print(f"{'n':>3}{'turn':>6}{'peak(m)':>9}{'peak_phase':>11}{'mid(m)':>8}{'recover(m)':>11}{'recover/side':>13}")
for n in NSIDES:
    tj=RegularPolygon(n,SIDE*n); prof=np.zeros(NB); cnt=np.zeros(NB)
    for mc in range(MC):
        er,ar=series(tj,int(np.random.SeedSequence([2033,n,N,int(TR*1000),mc]).generate_state(1)[0]))
        ph=(ar%SIDE)/SIDE; b=np.minimum((ph*NB).astype(int),NB-1)
        np.add.at(prof,b,er**2); np.add.at(cnt,b,1)
    prof=np.sqrt(prof/np.maximum(cnt,1))
    mid=float(prof[int(0.4*NB):int(0.6*NB)].mean()); ipk=int(np.argmax(prof)); peak=float(prof[ipk])
    rec=None
    for i in range(ipk,NB):
        if prof[i]<=1.1*mid: rec=(i+0.5)/NB*SIDE; break
    out[f"n{n}"]={'turn_deg':360/n,'side_m':SIDE,'peak_m':round(peak,3),'peak_phase':round((ipk+0.5)/NB,2),'mid_m':round(mid,3),
                  'recovery_m':None if rec is None else round(rec,2),'profile':[round(float(x),4) for x in prof]}
    print(f"{n:3d}{360/n:6.0f}{peak:9.3f}{(ipk+0.5)/NB:11.2f}{mid:8.3f}{'  n/a' if rec is None else f'{rec:11.2f}'}{'' if rec is None else f'{rec/SIDE:13.2f}'}")
json.dump({'config':{'side_m':SIDE,'n_sides':NSIDES,'N':N,'tr':TR,'MC':MC,'phase_bins':NB,'recovery_rule':'first phase after peak with error <= 1.1 x mid-side level'},'results':out},open('../data/corner_response.json','w'),indent=1)
print('-> ../data/corner_response.json')
