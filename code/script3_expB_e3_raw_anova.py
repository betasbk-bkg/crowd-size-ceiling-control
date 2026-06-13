"""
SCRIPT 3 — E3 Raw Runs + 실제 ANOVA
======================================
목적: 메이저 3 해결
     raw MC=15 runs 저장 → pseudo-obs 없이 실제 ANOVA

필요 파일:
  E3_supplement_proper.json  ← sanity check용

출력 파일:
  E3_raw_runs.json           ← raw runs 포함
  anova_raw_results.json     ← 실제 ANOVA 결과

실행:
  pip install pingouin
  python script3_expB_e3_raw_anova.py

설계:
  circle + square
  × N [5,10,15,20,25,30,40,50,75,100,150,200]
  × troll [5%, 15%, 40%]  — ANOVA 3-level
  × MC=15
  = 2 × 12 × 3 × 15 = 1,080 runs (~3분)

seed = i*31+N+int(tr*100)  (run_E2_E3_proper.py 동일)
"""
import numpy as np, json, time
from scipy.optimize import curve_fit

# ================================================================
# 원본 엔진 (run_E2_E3_proper.py 완전 동일)
# ================================================================
DT=1/60; SMOOTH=0.2; WIN=0.3; DELAY_F=26; DUR=65.0
FRAMES=int(DUR/DT); LOOK=2.0; VOTE_INT=int(WIN/DT)
S2=np.sqrt(2)/2
DIRS=np.array([[1,0],[S2,S2],[0,1],[-S2,S2],[-1,0],[-S2,-S2],[0,-1],[S2,-S2]])
DA=np.degrees(np.arctan2(DIRS[:,1],DIRS[:,0]))%360

def a2d(angles):
    a=angles%360; d=np.abs(DA[None,:]-a[:,None])
    d=np.minimum(d,360-d); return np.argmin(d,axis=1)

class Circle:
    def __init__(s,R=10): s.R=R
    def closest(s,p):
        t=np.arctan2(p[1],p[0]); cp=s.R*np.array([np.cos(t),np.sin(t)])
        return cp,(t%(2*np.pi))*s.R
    def at(s,arc): t=arc/s.R; return s.R*np.array([np.cos(t),np.sin(t)])
    def start(s): return np.array([s.R,0.])

class Square:
    def __init__(s,h=10):
        s.c=np.array([[h,0],[h,h],[-h,h],[-h,-h],[h,-h],[h,0.]],dtype=float)
        s.segs=[(s.c[i],s.c[i+1]) for i in range(5)]
        s.lens=[np.linalg.norm(b-a) for a,b in s.segs]
        s.circ=sum(s.lens); s.cum=np.array([0]+list(np.cumsum(s.lens)))
    def closest(s,p):
        bd,bp,ba=1e10,s.c[0],0.
        for i,(a,b) in enumerate(s.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2,0,1); pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,s.cum[i]+t*s.lens[i]
        return bp,ba
    def at(s,arc):
        arc=arc%s.circ
        for i,(a,b) in enumerate(s.segs):
            if arc<=s.cum[i+1]+1e-9:
                t=(arc-s.cum[i])/s.lens[i]; return a+np.clip(t,0,1)*(b-a)
        return s.c[-1]
    def start(s): return s.c[0].copy()

def gen_votes(iang,pang,tr,N,rng):
    sc=(1-tr)/0.95; na=round(N*0.70*sc); ns_=round(N*0.20*sc)
    nt=round(N*tr); no=max(0,N-na-ns_-nt)
    angs=np.empty(na+ns_+no); i=0
    angs[i:i+na]=iang+rng.uniform(-3,3,na); i+=na
    diff=iang-pang
    if diff>180: diff-=360
    if diff<-180: diff+=360
    if ns_>0: angs[i:i+ns_]=pang+diff*(1-rng.uniform(0.2,0.5,ns_)); i+=ns_
    if no>0: angs[i:i+no]=iang+rng.uniform(-30,30,no); i+=no
    votes=a2d(angs[:i])
    trolls=rng.integers(0,8,nt) if nt>0 else np.array([],dtype=int)
    return np.concatenate([votes,trolls])

def sim(traj,N_ag,tr,seed,speed=5.0):
    rng=np.random.default_rng(seed); pos=traj.start(); vel=np.zeros(2)
    pos_hist=[pos.copy()]; pang=0.; cur_dir=np.array([1.,0.]); errs=np.empty(FRAMES)
    for f in range(FRAMES):
        if f%VOTE_INT==0:
            di=max(0,len(pos_hist)-1-DELAY_F); dp=pos_hist[di]
            _,arc=traj.closest(dp); lap=traj.at(arc+LOOK)
            idir=lap-dp; n=np.linalg.norm(idir)
            if n>1e-10: idir/=n
            iang=np.degrees(np.arctan2(idir[1],idir[0]))
            votes=gen_votes(iang,pang,tr,N_ag,rng); pang=iang
            bl=DIRS[votes].mean(axis=0); cg=np.linalg.norm(bl)
            cur_dir=bl/cg if cg>1e-10 else np.array([1.,0.])
        vel+=SMOOTH*(cur_dir*speed-vel); pos=pos+vel*DT; pos_hist.append(pos.copy())
        cp,_=traj.closest(pos); errs[f]=np.linalg.norm(pos-cp)
    return float(np.sqrt(np.mean(errs**2)))

# ================================================================
# Sanity check — E3_supplement_proper.json 기준
# (troll=5%, 20%, 40%만 — 외부 파일 의존 없이)
# ================================================================
print("=== Sanity check vs E3_supplement_proper.json ===")
with open('E3_supplement_proper.json') as f: e3ref=json.load(f)['results']
c=Circle(); sq=Square()

checks=[
    (c,  50, 0.20, 'circle_tr0.20_N50'),
    (c,  5,  0.40, 'circle_tr0.40_N5'),
    (sq, 50, 0.20, 'square_tr0.20_N50'),
    (sq, 5,  0.05, 'square_tr0.05_N5'),
]
all_ok=True
for tobj,N,tr,key in checks:
    runs=[sim(tobj,N,tr,seed=i*31+N+int(tr*100)) for i in range(15)]
    got=round(np.mean(runs),4); exp=e3ref[key]['rmse_mean']
    diff=abs(got-exp)/exp*100; status='✓' if diff<0.1 else '✗'
    if status=='✗': all_ok=False
    print(f"  {status} {key}: got={got:.4f} ref={exp:.4f} diff={diff:.1f}%")
if not all_ok:
    print("  FAIL — 엔진 불일치. 중단."); exit(1)
print("  All PASS\n")

# ================================================================
# E3 Raw Runs 실험
# troll=[5%, 15%, 40%] — 3-level ANOVA 설계
# ================================================================
MC=15; V=5.0
NS     = [5,10,15,20,25,30,40,50,75,100,150,200]
TROLLS = [0.05, 0.15, 0.40]
TRAJS  = [('circle',c), ('square',sq)]
total  = len(NS)*len(TROLLS)*len(TRAJS)*MC

print(f"=== E3 Raw Runs | {total} runs | troll=[5%,15%,40%] | seed=i*31+N+int(tr*100) ===")
raw_results={}; t0=time.time(); done=0

for tname,tobj in TRAJS:
    for tr in TROLLS:
        for N in NS:
            rmses=[sim(tobj,N,tr,seed=i*31+N+int(tr*100)) for i in range(MC)]
            done+=MC
            key=f"{tname}_tr{tr:.2f}_N{N}"
            raw_results[key]={
                'rmse_mean': round(float(np.mean(rmses)),4),
                'rmse_std':  round(float(np.std(rmses)), 4),
                'rmse_ci95': round(float(1.96*np.std(rmses)/np.sqrt(MC)),4),
                'rmse_raw':  [round(float(r),5) for r in rmses],
                'mc_runs':   MC,
            }
            eta=(total-done)/(done/(time.time()-t0)) if done>0 else 0
            print(f"  {key}: {raw_results[key]['rmse_mean']:.4f}±{raw_results[key]['rmse_std']:.4f}"
                  f"  [{done}/{total} ETA {eta:.0f}s]")

elapsed_sim=time.time()-t0
print(f"\n시뮬레이션 완료: {elapsed_sim:.1f}s")

# ================================================================
# 실제 ANOVA (raw runs 기반)
# ================================================================
print("\n=== 2-way ANOVA on raw MC=15 runs ===")
try:
    import pingouin as pg
    import pandas as pd
except ImportError:
    print("pingouin 없음: pip install pingouin 후 재실행")
    # raw 파일은 저장
    out_raw={
        'config':{'MC':MC,'speed':V,'Ns':NS,'trolls':TROLLS,
                   'seed_pattern':'seed=i*31+N+int(tr*100)'},
        'results':raw_results,
        'metadata':{'elapsed_sec':round(elapsed_sim,1),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
    }
    with open('E3_raw_runs.json','w') as f: json.dump(out_raw,f,indent=2)
    print("E3_raw_runs.json 저장 완료. ANOVA는 pingouin 설치 후 재실행.")
    exit(0)

rows=[]
for key,val in raw_results.items():
    parts=key.split('_')
    traj=parts[0]; tr=float(parts[1].replace('tr','')); N=int(parts[2].replace('N',''))
    for rmse in val['rmse_raw']:
        rows.append({
            'trajectory': traj,
            'N_cat':      str(N),
            'troll_pct':  f"{int(tr*100)}%",
            'rmse':       float(rmse)
        })
df=pd.DataFrame(rows)

anova_results=[]
for traj in ['circle','square']:
    sub=df[df['trajectory']==traj].copy()
    aov=pg.anova(data=sub,dv='rmse',between=['N_cat','troll_pct'],detailed=True)
    ss_total=aov['SS'].sum()

    res={'trajectory':traj,
         'method':'raw_MC15_runs',
         'n_obs':len(sub),
         'design':f"{len(NS)}N × {len(TROLLS)}troll × MC={MC}"}

    label_map={
        'N_cat':            'N',
        'troll_pct':        'troll',
        'N_cat * troll_pct':'interaction',
        'Residual':         'residual'
    }
    for _,row in aov.iterrows():
        short=label_map.get(row['Source'], row['Source'])
        eta2=round((row['SS']/ss_total)*100, 3)
        p=row.get('p_unc', float('nan'))
        F=row.get('F',    float('nan'))
        df_=row.get('DF', float('nan'))
        res[f'eta2_{short}'] = eta2
        res[f'F_{short}']    = round(float(F),3)  if not (isinstance(F,  float) and np.isnan(F))  else None
        res[f'p_{short}']    = float(f"{p:.4e}")  if not (isinstance(p,  float) and np.isnan(p))  else None
        res[f'df_{short}']   = int(df_)            if not (isinstance(df_,float) and np.isnan(df_)) else None

    anova_results.append(res)
    print(f"\n  [{traj}] — RAW ANOVA (n={len(sub)})")
    print(f"    η²_N          : {res['eta2_N']:.1f}%  (F={res['F_N']}, p={res['p_N']})")
    print(f"    η²_troll      : {res['eta2_troll']:.1f}%  (F={res['F_troll']}, p={res['p_troll']})")
    print(f"    η²_interaction: {res['eta2_interaction']:.1f}%  (F={res['F_interaction']}, p={res['p_interaction']})")

# ================================================================
# 저장
# ================================================================
elapsed=time.time()-t0
out_raw={
    'config':{'MC':MC,'speed':V,'Ns':NS,'trolls':TROLLS,
               'engine':'run_E2_E3_proper.py',
               'seed_pattern':'seed=i*31+N+int(tr*100)',
               'note':'raw MC runs stored per condition'},
    'results':raw_results,
    'metadata':{'elapsed_sec':round(elapsed,1),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
}
with open('E3_raw_runs.json','w') as f: json.dump(out_raw,f,indent=2)

out_anova={
    'analysis_method':'2-way ANOVA (N × troll) — raw MC=15 runs',
    'data_source':'E3_raw_runs.json',
    'design':f"{len(NS)}N × {len(TROLLS)}troll × MC={MC} raw runs",
    'trajectories_analyzed':['circle','square'],
    'note':'No pseudo-observation reconstruction. Direct ANOVA on actual MC realizations.',
    'results':anova_results,
    'metadata':{'elapsed_sec':round(elapsed,1),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
}
with open('anova_raw_results.json','w') as f: json.dump(out_anova,f,indent=2)

print(f"\nSaved: E3_raw_runs.json, anova_raw_results.json  ({elapsed:.1f}s total)")
print("Done.")
