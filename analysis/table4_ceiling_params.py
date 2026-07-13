"""Fitted ceiling parameters (paper Table 4).
a,b via weighted least squares (weights 1/sem^2, sem = std/sqrt(50)) on cell means;
95% CIs via known-variance GLS covariance (X'WX)^-1 with z = 1.96;
attainable reduction = (b/sqrt(5)) / (a + b/sqrt(5)).
Verified 2026-07-12: reproduces all 6 rows of Table 4 (a to 4 dp, b to 3 dp, CIs, attainable).
"""
import json, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
D=json.load(open(os.path.join(ROOT,'data/campaign_main_mc50_fixed.json')))['results']
NS=np.array([5,10,15,20,25,30,40,50,75,100,150,200],float)
X=np.column_stack([np.ones_like(NS),1/np.sqrt(NS)])
for tj,tr in [('circle',0.05),('circle',0.20),('circle',0.40),('zigzag',0.05),('zigzag',0.20),('zigzag',0.40)]:
    y=np.array([D[f"{tj}_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean'] for N in NS])
    sem=np.array([D[f"{tj}_tr{tr:.2f}_N{int(N)}_T0"]['rmse_std'] for N in NS])/np.sqrt(50)
    W=np.diag(1/sem**2); XtWX=X.T@W@X
    a,b=np.linalg.solve(XtWX,X.T@W@y)
    se=np.sqrt(np.diag(np.linalg.inv(XtWX)))
    att=100*(b/np.sqrt(5))/(a+b/np.sqrt(5))
    print(f"{tj:7s} tr={tr:.0%}: a={a:.4f} [{a-1.96*se[0]:.3f},{a+1.96*se[0]:.3f}]  "
          f"b={b:+.3f} [{b-1.96*se[1]:+.3f},{b+1.96*se[1]:+.3f}]  attainable={att:.2f}%")
