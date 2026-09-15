#!/usr/bin/env python3
"""
anova_raw_mc15.py — regenerate the MC = 15 two-way ANOVA table from archived raw runs.

The original campaign script (code/script3_expB_e3_raw_anova.py) simulates and analyses in one
pass, writing E3_raw_runs.json and anova_raw_results.json. Only the raw runs were archived, so
this script reproduces the analysis half from them. The MC = 15 values it returns are the ones
quoted in the manuscript alongside the MC = 50 results (eta^2 interaction 23.3% circle,
9.4% square).

READS : ../data/E3_raw_runs.json      WRITES: ../data/anova_raw_results.json
RUN   : cd analysis ; python anova_raw_mc15.py      (seconds)
"""
import json, time, numpy as np
from scipy import stats

D = json.load(open('../data/E3_raw_runs.json'))
cfg, R = D['config'], D['results']
NS, TRS, MC = cfg['Ns'], cfg['trolls'], cfg['MC']
out = []
print(f"{'traj':<10}{'eta2_N':>9}{'eta2_tr':>9}{'eta2_int':>10}{'eta2_res':>10}")
for traj in ['circle', 'square']:
    cell = {(N, tr): np.asarray(R[f"{traj}_tr{tr:.2f}_N{N}"]['rmse_raw'], float) for tr in TRS for N in NS}
    allv = np.concatenate(list(cell.values())); gm = allv.mean()
    a, b, n = len(NS), len(TRS), MC
    ssA = sum(b*n*(np.mean([cell[(N, tr)].mean() for tr in TRS])-gm)**2 for N in NS)
    ssB = sum(a*n*(np.mean([cell[(N, tr)].mean() for N in NS])-gm)**2 for tr in TRS)
    ssT = ((allv-gm)**2).sum(); ssC = sum(n*(v.mean()-gm)**2 for v in cell.values())
    ssAB = ssC-ssA-ssB; ssE = ssT-ssC
    dfA, dfB = a-1, b-1; dfAB = dfA*dfB; dfE = a*b*(n-1)
    F = [(ssA/dfA)/(ssE/dfE), (ssB/dfB)/(ssE/dfE), (ssAB/dfAB)/(ssE/dfE)]
    P = [stats.f.sf(F[0], dfA, dfE), stats.f.sf(F[1], dfB, dfE), stats.f.sf(F[2], dfAB, dfE)]
    eta = [100*ssA/ssT, 100*ssB/ssT, 100*ssAB/ssT, 100*ssE/ssT]
    res = {'trajectory': traj, 'method': 'raw_MC15_runs', 'n_obs': int(allv.size),
           'design': f"{a}N x {b}tr x MC={MC}"}
    for k, i, df in [('N', 0, dfA), ('troll', 1, dfB), ('interaction', 2, dfAB)]:
        res[f'eta2_{k}'] = round(eta[i], 3); res[f'F_{k}'] = round(float(F[i]), 3)
        res[f'p_{k}'] = float(f"{P[i]:.4e}"); res[f'df_{k}'] = int(df)
    res['eta2_residual'] = round(eta[3], 3); res['df_residual'] = int(dfE)
    out.append(res)
    print(f"{traj:<10}{eta[0]:8.1f}%{eta[1]:8.1f}%{eta[2]:9.1f}%{eta[3]:9.1f}%")
json.dump({'analysis_method': '2-way ANOVA (N x troll) - raw MC=15 runs',
           'data_source': 'E3_raw_runs.json',
           'design': f"{len(NS)}N x {len(TRS)}troll x MC={MC} raw runs",
           'trajectories_analysed': ['circle', 'square'],
           'note': 'No pseudo-observation reconstruction. Direct ANOVA on the archived MC realisations.',
           'results': out,
           'metadata': {'regenerated_from': 'E3_raw_runs.json', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
          open('../data/anova_raw_results.json', 'w'), indent=2)
print('-> ../data/anova_raw_results.json')
