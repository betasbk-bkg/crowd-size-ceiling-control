"""Two-way ANOVA + assumption diagnostics (paper Table 2 and its caption).
Raw two-way ANOVA (N x tr, n=50/cell) on per-run RMSE; eta^2 = SS/SS_total*100
(residual row reported as 100 - sum of the three effects in the paper).
Diagnostics: Shapiro-Wilk on pooled cell-centered residuals; Levene (median-centered,
scipy default) across the 36 cells; max/min cell-variance ratio; residual skew.
Verified 2026-07-12: eta^2/F match Table 2 to reported precision;
p-values circle (tr 2.7e-277, int 8.9e-248; N underflow < 1e-300),
square (N 2.5e-189, tr 5.5e-184, int 5.7e-68); Levene circle 2.0e-72, square 2.1e-60 (< 1e-59).
"""
import json, numpy as np, os
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
d=json.load(open(os.path.join(ROOT,'data/anova_raw_mc50.json')))
r=d['results']; NS=d['config']['Ns']; TRS=d['config']['trolls']
for tj in ['circle','square']:
    data={(N,tr):np.array(r[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_raw']) for tr in TRS for N in NS}
    allv=np.concatenate(list(data.values())); gm=allv.mean(); a,b,n=len(NS),len(TRS),50
    ssA=sum(b*n*(np.mean([data[(N,tr)].mean() for tr in TRS])-gm)**2 for N in NS)
    ssB=sum(a*n*(np.mean([data[(N,tr)].mean() for N in NS])-gm)**2 for tr in TRS)
    ssT=((allv-gm)**2).sum()
    ssC=sum(n*(v.mean()-gm)**2 for v in data.values()); ssAB=ssC-ssA-ssB; ssE=ssT-ssC
    dfA,dfB=a-1,b-1; dfAB=dfA*dfB; dfE=a*b*(n-1); ms=lambda ss,df: ss/df
    F=[ms(ssA,dfA)/ms(ssE,dfE), ms(ssB,dfB)/ms(ssE,dfE), ms(ssAB,dfAB)/ms(ssE,dfE)]
    p=[stats.f.sf(F[0],dfA,dfE), stats.f.sf(F[1],dfB,dfE), stats.f.sf(F[2],dfAB,dfE)]
    resid=np.concatenate([v-v.mean() for v in data.values()])
    print(f"== {tj} ==")
    print(f" eta2: N={100*ssA/ssT:.1f}% tr={100*ssB/ssT:.1f}% int={100*ssAB/ssT:.1f}% "
          f"(residual as reported = {100-round(100*ssA/ssT,1)-round(100*ssB/ssT,1)-round(100*ssAB/ssT,1):.1f}%)")
    print(f" F: N={F[0]:.1f} tr={F[1]:.1f} int={F[2]:.1f} | p: N={p[0]:.2e} tr={p[1]:.2e} int={p[2]:.2e}")
    print(f" Shapiro p={stats.shapiro(resid).pvalue:.2e}  Levene p={stats.levene(*data.values()).pvalue:.2e}")
    vs=[v.var(ddof=1) for v in data.values()]
    print(f" var-ratio={max(vs)/min(vs):.1f}  skew={stats.skew(resid):+.3f}")
