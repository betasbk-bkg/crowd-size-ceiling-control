"""BUG-CELL FIX rerun — replicates the 2026-07 session fix exactly.
Fix: nt=round(N*tr) canonical; honest remainder split 70/95:20/95 with negative clamp into 'accurate'.
Seed: SeedSequence([2026, traj_id, N, tr_milli, model_id, 0, 0, mc])  (July-session convention)
"""
import sys, json, time
import numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,os.path.join(ROOT,'code'))
import adversary_ladder as al

def _honest_block_fixed(iang, pang, tr, N, rng):
    nt = round(N * tr); nt = min(nt, N)
    rem = N - nt
    na = round(rem * 0.7368); ns = round(rem * 0.2105)
    no = rem - na - ns
    if no < 0: na += no; no = 0
    angs = np.empty(na+ns+no); i=0
    angs[i:i+na] = iang + rng.uniform(-3,3,na); i+=na
    diff = iang - pang
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if ns>0: angs[i:i+ns] = pang + diff*(1-rng.uniform(0.2,0.5,ns)); i+=ns
    if no>0: angs[i:i+no] = iang + rng.uniform(-30,30,no); i+=no
    assert i == rem, f'composition mismatch: {i}!={rem}'
    return al.a2d(angs[:i]), nt
al._honest_block = _honest_block_fixed

MC=50; N=5; TR=0.30; SPEED=5.0; K_T2=10
TRAJS={'circle':al.Circle(),'square':al.Square(),'lemniscate':al.Lemniscate(),'zigzag':al.Zigzag()}
TRAJ_ID={'circle':0,'square':1,'lemniscate':2,'zigzag':3}
MODEL_ID={'T0':0,'T1':1,'T2':2}
def make_seed(tj,model,mc):
    return int(np.random.SeedSequence([2026,TRAJ_ID[tj],N,int(round(TR*1000)),MODEL_ID[model],0,0,mc]).generate_state(1)[0])

# sanity print
sc=(1-TR)/0.95
print(f"BEFORE: na={round(N*0.70*sc)} ns={round(N*0.20*sc)} nt={round(N*TR)} total={round(N*0.70*sc)+round(N*0.20*sc)+round(N*TR)}")
rem=N-round(N*TR); na1=round(rem*0.7368); ns1=round(rem*0.2105); no1=rem-na1-ns1
if no1<0: na1+=no1; no1=0
print(f"AFTER : na={na1} ns={ns1} no={no1} nt={round(N*TR)} total={na1+ns1+no1+round(N*TR)}")

t0=time.time(); out={}
for tj,tobj in TRAJS.items():
    for model in ['T0','T1','T2']:
        rmses=[]; gammas=[]; seeds=[]
        for mc in range(MC):
            s=make_seed(tj,model,mc); seeds.append(s)
            if model=='T0': r=al.sim(tobj,N,TR,seed=s,model='T1',coherence=0.0,speed=SPEED)
            elif model=='T1': r=al.sim(tobj,N,TR,seed=s,model='T1',coherence=1.0,speed=SPEED)
            else: r=al.sim_ext(tobj,N,TR,seed=s,model='T2',coherence=1.0,speed=SPEED,k_hold=K_T2)
            rmses.append(r['rmse']); gammas.append(r.get('gm', r.get('gamma_mean',np.nan)))
        a=np.array(rmses)
        out[f"{tj}_tr0.30_N5_{model}"]={'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
            'rmse_ci95':round(float(1.96*a.std()/np.sqrt(MC)),4),'gamma_mean':round(float(np.nanmean(gammas)),4),
            'mc_runs':MC,'seed_first':seeds[0]}
        print(f"{tj:11s} {model}: rmse={a.mean():.4f}  [{time.time()-t0:.0f}s]")
json.dump(out, open(os.path.join(ROOT,'data/bugfix_N5_tr30_results.json'),'w'), indent=1)
print(f"done {time.time()-t0:.0f}s")
