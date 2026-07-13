"""Lag-axis sensitivity regeneration (Table 3): tr=20%, T0, MC=50.
Conditions: (alpha, tau_frames) in {(0.20,13),(0.10,13),(0.20,26),(0.10,26)} x {lemniscate,zigzag} x N in {5,200}.
Uses bug-fixed honest block (irrelevant at N=5 tr=20: no overflow, but applied for consistency).
Seed: SeedSequence([2026, 9, traj_id, N, alpha_pct, tau, mc])  (this-session convention, documented)
"""
import sys, json, time, numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,os.path.join(ROOT,'code'))
import adversary_ladder as al
# bug-fixed honest block (same as bugfix_N5_tr30_rerun.py)
def _hb(iang,pang,tr,N,rng):
    nt=round(N*tr); nt=min(nt,N); rem=N-nt
    na=round(rem*0.7368); ns=round(rem*0.2105); no=rem-na-ns
    if no<0: na+=no; no=0
    angs=np.empty(na+ns+no); i=0
    angs[i:i+na]=iang+rng.uniform(-3,3,na); i+=na
    diff=iang-pang
    if diff>180: diff-=360
    if diff<-180: diff+=360
    if ns>0: angs[i:i+ns]=pang+diff*(1-rng.uniform(0.2,0.5,ns)); i+=ns
    if no>0: angs[i:i+no]=iang+rng.uniform(-30,30,no); i+=no
    return al.a2d(angs[:i]), nt
al._honest_block=_hb

MC=50; TR=0.20; SPEED=5.0
CONDS=[(0.20,13),(0.10,13),(0.20,26),(0.10,26)]
TRAJS={'lemniscate':al.Lemniscate(),'zigzag':al.Zigzag()}
TID={'lemniscate':2,'zigzag':3}
out={}; t0=time.time()
for (aA,tF) in CONDS:
    al.SMOOTH=aA; al.DELAY_F=tF
    for tj,tobj in TRAJS.items():
        samp={}
        for N in [5,200]:
            rs=[]
            for mc in range(MC):
                s=int(np.random.SeedSequence([2026,9,TID[tj],N,int(aA*100),tF,mc]).generate_state(1)[0])
                rs.append(al.sim(tobj,N,TR,seed=s,model='T1',coherence=0.0,speed=SPEED)['rmse'])
            samp[N]=rs
        m5,m200=np.mean(samp[5]),np.mean(samp[200])
        from scipy import stats
        t,p=stats.ttest_ind(samp[5],samp[200],equal_var=False)
        d=100*(m5-m200)/m5
        out[f"{tj}_a{aA}_t{tF}"]={'rmse5':round(m5,4),'rmse200':round(m200,4),'delta_pct':round(d,2),'welch_p':float(p),'raw5':samp[5],'raw200':samp[200]}
        print(f"a={aA} tau={tF} {tj:11s}: Δ={d:+.2f}%  p={p:.1e}  [{time.time()-t0:.0f}s]")
json.dump(out,open(os.path.join(ROOT,'data/lagaxis_results.json'),'w'))
print("done",round(time.time()-t0),"s")
