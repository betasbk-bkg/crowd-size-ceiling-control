"""Quadrature-model calibration (paper Discussion, 'Origin of the transition region').
Model: e^2 = e_lag^2 + e_adv^2, e_lag = kappa_g*v*tau_eff, e_adv = c_a*tr*v*T_c/sqrt(N).
Constants: v=5, tau=26/60 s, T_v=0.3 s, alpha=0.2, dt=1/60;
tau_eff = tau - dt/ln(1-alpha) = 0.508 s, T_c = tau + T_v/2 = 0.583 s.
Fit: nonlinear least squares over the full circle grid (120 cells, corrected data).
Verified 2026-07-12: kappa_g=0.319, c_a=1.67, R^2=0.889;
implied-b/empirical-b ratios: tr=40% 1.13, tr=5% 0.40, tr=20% 0.71.
"""
import json, numpy as np, os
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
D=json.load(open(os.path.join(ROOT,'data/campaign_main_mc50_fixed.json')))['results']
NS=np.array([5,10,15,20,25,30,40,50,75,100,150,200],float)
TRS=np.array([0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40])
v=5.0; tau=26/60; Tv=0.3; dt=1/60; alpha=0.2
tau_eff=tau-dt/np.log(1-alpha); Tc=tau+Tv/2
Y,TR,NN=[],[],[]
for tr in TRS:
    for N in NS:
        Y.append(D[f"circle_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean']); TR.append(tr); NN.append(N)
Y,TR,NN=map(np.array,(Y,TR,NN))
f=lambda p: np.sqrt((p[0]*v*tau_eff)**2+(p[1]*TR*v*Tc/np.sqrt(NN))**2)
r=least_squares(lambda p: f(p)-Y,[0.3,1.5]); kg,ca=r.x
R2=1-np.sum((Y-f(r.x))**2)/np.sum((Y-Y.mean())**2)
print(f"tau_eff={tau_eff:.4f}s T_c={Tc:.4f}s | kappa_g={kg:.3f} c_a={ca:.2f} R^2={R2:.3f}")
X=np.column_stack([np.ones_like(NS),1/np.sqrt(NS)])
for tr in [0.40,0.05,0.20]:
    y_emp=np.array([D[f"circle_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean'] for N in NS])
    b_emp=np.linalg.lstsq(X,y_emp,rcond=None)[0][1]
    y_mod=np.sqrt((kg*v*tau_eff)**2+(ca*tr*v*Tc/np.sqrt(NS))**2)
    b_mod=np.linalg.lstsq(X,y_mod,rcond=None)[0][1]
    print(f"tr={tr:.0%}: implied-b/empirical-b = {b_mod/b_emp:.2f}")
