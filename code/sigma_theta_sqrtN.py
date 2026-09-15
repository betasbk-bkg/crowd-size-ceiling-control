#!/usr/bin/env python3
"""
sigma_theta_sqrtN.py — consensus-direction noise vs crowd size (vote-level, no dynamics).

Supports the Methods sentence: "the standard deviation of the consensus direction times sqrt(N)
is constant across N = 5..800 and the mean bias is below 0.1 degrees". Uses the engine's own
vote generator (honest block + uniform adversaries, T1 c=0 == T0) at a fixed target direction.

READS : engine only     WRITES: ../data/sigma_theta_sqrtN.json
RUN   : cd code ; python sigma_theta_sqrtN.py     (~1 min)
Seed  : SeedSequence([2031, N, int(tr*1000)])  — 20,000 vote rounds per cell
"""
import json, numpy as np, adversary_ladder as E
NS=[5,10,20,50,100,200,400,800]; TRS=[0.05,0.20,0.40]; DRAWS=20000
# Bias depends on where the target sits relative to the 8 vote directions: it is zero on an axis
# (target 0 deg) and up to 22.5 deg midway between axes (honest quantisation, independent of N).
# The Methods claim "bias below 0.1 deg" is the on-axis / direction-averaged statement; both are
# reported: 'axis' = target 0 deg, 'random' = target drawn uniformly per draw (path-averaged case).
out={}
print(f"{'tr':>5}{'N':>6}{'sd(deg)':>10}{'sd*sqrtN':>10}{'bias axis':>11}{'bias rand':>11}")
for tr in TRS:
    for N in NS:
        rng=np.random.default_rng(int(np.random.SeedSequence([2031,N,int(tr*1000)]).generate_state(1)[0]))
        dev=np.empty(DRAWS); devr=np.empty(DRAWS); pang=0.0
        for i in range(DRAWS):
            v=E.gen_votes_adv(0.0,pang,tr,N,rng,'T1',0.0,None)
            bl=E.DIRS[v].mean(axis=0); dev[i]=(np.degrees(np.arctan2(bl[1],bl[0]))+180)%360-180
            ia=rng.uniform(0,360); v=E.gen_votes_adv(ia,ia,tr,N,rng,'T1',0.0,None)
            bl=E.DIRS[v].mean(axis=0); devr[i]=(np.degrees(np.arctan2(bl[1],bl[0]))-ia+180)%360-180
        sd=float(dev.std()); bias=float(dev.mean()); biasr=float(devr.mean())
        out[f"tr{tr:.2f}_N{N}"]={'sd_deg':round(sd,3),'sd_sqrtN':round(sd*np.sqrt(N),2),'bias_axis_deg':round(bias,4),'bias_random_target_deg':round(biasr,4),'draws':DRAWS}
        print(f"{tr:5.0%}{N:6d}{sd:10.3f}{sd*np.sqrt(N):10.2f}{bias:11.4f}{biasr:11.4f}")
summ={}
for tr in TRS:
    vals=[out[f"tr{tr:.2f}_N{N}"]['sd_sqrtN'] for N in NS if N>=20]; b=[abs(out[f"tr{tr:.2f}_N{N}"]['bias_axis_deg']) for N in NS]; br=[abs(out[f"tr{tr:.2f}_N{N}"]['bias_random_target_deg']) for N in NS]
    summ[f"tr{tr:.2f}"]={'sd_sqrtN_mean_N>=20':round(float(np.mean(vals)),2),'sd_sqrtN_min':min(vals),'sd_sqrtN_max':max(vals),'max_abs_bias_axis_deg':round(max(b),4),'max_abs_bias_random_deg':round(max(br),4)}
    print(f"  tr={tr:.0%}: sd*sqrtN (N>=20) = {np.mean(vals):.1f} (range {min(vals)}-{max(vals)}), max |bias| axis {max(b):.4f} / random-target {max(br):.4f} deg")
json.dump({'config':{'Ns':NS,'trolls':TRS,'draws':DRAWS,'target':'0 deg (axis) and uniform random per draw','model':'T1 c=0 (uniform)'},'results':out,'summary':summ},open('../data/sigma_theta_sqrtN.json','w'),indent=1)
print('-> ../data/sigma_theta_sqrtN.json')
