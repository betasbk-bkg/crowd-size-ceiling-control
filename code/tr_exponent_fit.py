#!/usr/bin/env python3
"""
tr_exponent_fit.py — quadrature model with a free adversarial exponent (analysis only).

Model: e^2 = (kappa_g v tau_eff)^2 + (c_a tr^p v T_c / sqrt(N))^2, fitted per geometry on the
corrected 120-cell grid (10 tr x 12 N, T0). p = 1 is the published form; p = 0.5 is the pure
random-vector-sum prediction. Reports p, R^2 and delta AIC (free p minus p = 1).

READS : ../data/campaign_main_mc50_fixed_zzfix.json     WRITES: ../data/tr_exponent_fit.json
RUN   : cd code ; python tr_exponent_fit.py      (seconds)
"""
import json, numpy as np
from scipy.optimize import least_squares
D=json.load(open('../data/campaign_main_mc50_fixed_zzfix.json'))['results']
NS=np.array([5,10,15,20,25,30,40,50,75,100,150,200],float); TRS=[0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40]
v=5.0; tau=26/60; Tv=0.3; dt=1/60; alpha=0.2; tau_eff=tau-dt/np.log(1-alpha); Tc=tau+Tv/2
out={}
print(f"{'geom':<12}{'p':>7}{'R2(free)':>10}{'R2(p=1)':>9}{'R2(p=.5)':>10}{'dAIC free-1':>12}{'dAIC .5-1':>10}")
for g in ['circle','square','lemniscate','zigzag']:
    Y=[];TR=[];NN=[]
    for tr in TRS:
        for N in NS: Y.append(D[f"{g}_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean']); TR.append(tr); NN.append(N)
    Y,TR,NN=map(np.array,(Y,TR,NN)); n=len(Y)
    def f(p,pw=None):
        pp=p[2] if pw is None else pw
        return np.sqrt((p[0]*v*tau_eff)**2+(p[1]*TR**pp*v*Tc/np.sqrt(NN))**2)
    def fit(pw):
        if pw is None: r=least_squares(lambda p: f(p)-Y,[0.3,1.5,1.0],bounds=([0,0,0.05],[5,50,3])); k=3
        else: r=least_squares(lambda p: f(p,pw)-Y,[0.3,1.5]); k=2
        res=Y-(f(r.x) if pw is None else f(r.x,pw)); r2=1-np.sum(res**2)/np.sum((Y-Y.mean())**2)
        aic=n*np.log(np.sum(res**2)/n)+2*k; return r.x,r2,aic
    xf,r2f,af=fit(None); x1,r21,a1=fit(1.0); xh,r2h,ah=fit(0.5)
    out[g]={'p_free':round(float(xf[2]),3),'kappa_g':round(float(xf[0]),3),'c_a':round(float(xf[1]),3),'R2_free':round(float(r2f),3),
            'R2_p1':round(float(r21),3),'R2_p05':round(float(r2h),3),'dAIC_free_minus_p1':round(float(af-a1),1),'dAIC_p05_minus_p1':round(float(ah-a1),1)}
    print(f"{g:<12}{xf[2]:7.3f}{r2f:10.3f}{r21:9.3f}{r2h:10.3f}{af-a1:12.1f}{ah-a1:10.1f}")
json.dump({'config':{'model':'e^2=(kg v tau_eff)^2+(ca tr^p v Tc/sqrtN)^2','tau_eff':tau_eff,'Tc':Tc,'data':'campaign_main_mc50_fixed_zzfix.json T0 120 cells'},'results':out},open('../data/tr_exponent_fit.json','w'),indent=1)
print('-> ../data/tr_exponent_fit.json')
