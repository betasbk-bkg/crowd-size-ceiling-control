"""Ceiling-fit R^2 grid (paper Table 1 + Supplementary Table S6).
Model: RMSE(N) = a + b/sqrt(N), unweighted OLS on MC=50 cell means, adversary T0.
Data: data/campaign_main_mc50_fixed.json (N=5, tr=30% composition-corrected).
Output: prints the 4x10 grid; writes data/supplement_ceiling_fits_tenlevel.csv.
Verified 2026-07-12: reproduces Table 1 (six-level subset) to 3 decimals, 24/24 cells.
"""
import json, csv, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
D=json.load(open(os.path.join(ROOT,'data/campaign_main_mc50_fixed.json')))['results']
NS=np.array([5,10,15,20,25,30,40,50,75,100,150,200],float)
TRS=[0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40]
X=np.column_stack([np.ones_like(NS),1/np.sqrt(NS)])
rows=[]
print(f"{'traj':11s}"+''.join(f" tr{int(t*100):02d}" for t in TRS))
for tj in ['circle','square','lemniscate','zigzag']:
    r2s=[]
    for tr in TRS:
        y=np.array([D[f"{tj}_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean'] for N in NS])
        beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        r2s.append(round(1-np.sum((y-X@beta)**2)/np.sum((y-y.mean())**2),3))
    rows.append([tj]+r2s)
    print(f"{tj:11s} "+' '.join(f"{v:.3f}" for v in r2s))
with open(os.path.join(ROOT,'data/supplement_ceiling_fits_tenlevel.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['trajectory']+[f"tr={int(t*100)}%" for t in TRS]); w.writerows(rows)
print("wrote data/supplement_ceiling_fits_tenlevel.csv")
