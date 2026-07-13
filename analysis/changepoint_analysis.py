"""Change-point on b(tr) (paper Results, transition region).
b(tr): ceiling fit per tr (curve_fit, cell means, T0). Two-segment fit: two linear
segments sharing the breakpoint sample (x[:bp+1] / x[bp:]), breakpoint minimizing
total SSE over the interior grid. 95% CI: parametric bootstrap (B=400) resampling
per-cell means with their Monte Carlo standard errors, seed=42.
Verified 2026-07-12 (corrected data): circle 17% CI[17,30], square 13% CI[13,17],
lemniscate 17% (bootstrap concentrated on the 17% grid level).
"""
import json, numpy as np, os
from scipy import optimize
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
D=json.load(open(os.path.join(ROOT,'data/campaign_main_mc50_fixed.json')))['results']
NS=[5,10,15,20,25,30,40,50,75,100,150,200]
TRS=np.array([0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40])
cm=lambda N,a,b: a+b/np.sqrt(N)
def bvec(tj,ym=None):
    out=[]
    for j,tr in enumerate(TRS):
        y=ym[j] if ym is not None else np.array([D[f'{tj}_tr{tr:.2f}_N{N}_T0']['rmse_mean'] for N in NS])
        out.append(optimize.curve_fit(cm,NS,y,p0=[y.min(),0.1])[0][1])
    return np.array(out)
def cp(bs):
    best=None
    for bp in range(2,len(TRS)-2):
        A1=np.polyfit(TRS[:bp+1],bs[:bp+1],1); A2=np.polyfit(TRS[bp:],bs[bp:],1)
        sse=np.sum((np.polyval(A1,TRS[:bp+1])-bs[:bp+1])**2)+np.sum((np.polyval(A2,TRS[bp:])-bs[bp:])**2)
        if best is None or sse<best[1]: best=(TRS[bp],sse)
    return best[0]
rng=np.random.default_rng(42)
for tj in ['circle','square','lemniscate']:
    pt=cp(bvec(tj))
    base=[np.array([D[f'{tj}_tr{tr:.2f}_N{N}_T0']['rmse_mean'] for N in NS]) for tr in TRS]
    sems=[np.array([D[f'{tj}_tr{tr:.2f}_N{N}_T0']['rmse_std'] for N in NS])/np.sqrt(50) for tr in TRS]
    boots=[cp(bvec(tj,[base[j]+rng.normal(0,sems[j]) for j in range(len(TRS))])) for _ in range(400)]
    lo,hi=np.percentile(boots,[2.5,97.5])
    print(f"{tj:11s} change-point={pt:.2f}  bootstrap 95% CI=[{lo:.2f},{hi:.2f}]")
