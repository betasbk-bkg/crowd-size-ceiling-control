"""
EXP-A: E2f MC=15 재실험
==========================
목적: 기존 part1_e2f.json (MC=10)을 MC=15로 전체 재실행
     → MC 통일 (전 실험 MC=15)
     → 메이저 4 완전 해결

설계: lemniscate + zigzag
      × N [5,10,20,30,50,75,100,150,200] (기존 9개 — 일관성 유지)
      × troll [5,10,20,30,40%]
      × MC=15
      → 2×9×5×15 = 1,350 runs

엔진: paper_fullscale.py 완전 동일
      seed = i*31 + N  (run_condition 원본)

Sanity check: 기존 part1_e2f.json 대비 diff < 3%

Output: E2f_mc15.json

실행:
  python expA_e2f_mc15.py
"""
import numpy as np, json, time
from scipy.optimize import curve_fit

# ================================================================
# 원본 엔진 (paper_fullscale.py 완전 동일)
# ================================================================
DT=1/60; SMOOTH=0.2; WIN=0.3; DELAY_F=26; DUR=65.0
FRAMES=int(DUR/DT); LOOK=2.0; VOTE_INT=int(WIN/DT)
S2=np.sqrt(2)/2
DIRS=np.array([[1,0],[S2,S2],[0,1],[-S2,S2],[-1,0],[-S2,-S2],[0,-1],[S2,-S2]])
DA=np.degrees(np.arctan2(DIRS[:,1],DIRS[:,0]))%360

def a2d(angles):
    a=angles%360; d=np.abs(DA[None,:]-a[:,None])
    d=np.minimum(d,360-d); return np.argmin(d,axis=1)

class Lemniscate:
    def __init__(self,a=7,n=800):
        self.name='lemniscate'
        ts=np.linspace(0,2*np.pi,n,endpoint=False)
        sn,cs=np.sin(ts),np.cos(ts); den=1+sn**2
        self.pts=np.column_stack([a*cs/den, a*sn*cs/den])
        dl=np.linalg.norm(np.diff(self.pts,axis=0),axis=1)
        self.arcs=np.concatenate([[0],np.cumsum(dl)]); self.circ=self.arcs[-1]
    def closest(self,p):
        d=np.linalg.norm(self.pts-p,axis=1); i=int(np.argmin(d))
        return self.pts[i].copy(),self.arcs[i]
    def at(self,arc):
        a=arc%self.circ; i=min(int(np.searchsorted(self.arcs,a)),len(self.pts)-1)
        return self.pts[i].copy()
    def start(self): return self.pts[0].copy()

class Zigzag:
    def __init__(self,amp=5,ns=10,sx=5):
        self.name='zigzag'; pts=[np.array([0.,0.])]
        for i in range(ns): pts.append(np.array([(i+1)*sx, amp if i%2==0 else 0.]))
        self.c=np.array(pts)
        self.segs=[(self.c[i],self.c[i+1]) for i in range(ns)]
        self.lens=[np.linalg.norm(b-a) for a,b in self.segs]
        self.circ=sum(self.lens); self.cum=np.array([0]+list(np.cumsum(self.lens)))
    def closest(self,p):
        bd,bp,ba=1e10,self.c[0],0.
        for i,(a,b) in enumerate(self.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2,0,1); pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba
    def at(self,arc):
        arc=arc%self.circ
        for i,(a,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9:
                t=(arc-self.cum[i])/self.lens[i]; return a+np.clip(t,0,1)*(b-a)
        return self.c[-1]
    def start(self): return self.c[0].copy()

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
# Sanity check (seed=i*31+N, MC=10 재현)
# ================================================================
print("=== Sanity check vs part1_e2f.json (MC=10, seed=i*31+N) ===")
with open('part1_e2f.json') as f: ref=json.load(f)['e2f_results']
lem=Lemniscate(); zz=Zigzag()

checks=[('lemniscate',lem,50, 0.20,'lemniscate_tr0.20_N50'),
        ('lemniscate',lem,200,0.40,'lemniscate_tr0.40_N200'),
        ('zigzag',    zz, 50, 0.20,'zigzag_tr0.20_N50'),
        ('zigzag',    zz, 100,0.40,'zigzag_tr0.40_N100')]
all_ok=True
for tname,tobj,N,tr,key in checks:
    runs=[sim(tobj,N,tr,seed=i*31+N) for i in range(10)]
    got=round(np.mean(runs),4); exp=ref[key]['rmse_m']
    diff=abs(got-exp)/exp*100; status='✓' if diff<2.0 else '✗'
    if status=='✗': all_ok=False
    print(f"  {status} {key}: got={got:.4f} ref={exp:.4f} diff={diff:.1f}%")
if not all_ok:
    print("  FAIL. 중단."); exit(1)
print("  All PASS\n")

# ================================================================
# E2f MC=15 전체 재실험
# seed = i*31 + N  (원본 run_condition 동일)
# ================================================================
MC=15; V=5.0
NS    = [5,10,20,30,50,75,100,150,200]   # 기존 e2f와 동일 9개
TROLLS= [0.05,0.10,0.20,0.30,0.40]
TRAJS = [('lemniscate',lem),('zigzag',zz)]
total = len(NS)*len(TROLLS)*len(TRAJS)*MC

print(f"=== E2f MC=15 재실험 | {total} runs | seed=i*31+N ===")
results={}; t0=time.time(); done=0

for tname,tobj in TRAJS:
    for tr in TROLLS:
        for N in NS:
            rmses=[sim(tobj,N,tr,seed=i*31+N) for i in range(MC)]
            done+=MC
            key=f"{tname}_tr{tr:.2f}_N{N}"
            results[key]={
                'rmse_mean': round(float(np.mean(rmses)),4),
                'rmse_std':  round(float(np.std(rmses)), 4),
                'rmse_ci95': round(float(1.96*np.std(rmses)/np.sqrt(MC)),4),
                'rmse_raw':  [round(float(r),5) for r in rmses],  # raw runs 저장
                'mc_runs':   MC,
            }
            eta=(total-done)/(done/(time.time()-t0)) if done>0 else 0
            print(f"  {key}: {results[key]['rmse_mean']:.4f}±{results[key]['rmse_std']:.4f}"
                  f"  [{done}/{total} ETA {eta:.0f}s]")

# ================================================================
# MC=10 vs MC=15 차이 검증
# ================================================================
def cm(N,a,b): return a+b/np.sqrt(N)
print("\n=== MC=10 vs MC=15 차이 검증 ===")
max_diff=0
for key,val in results.items():
    old_key = key  # 같은 key
    if old_key in ref:
        old_v = ref[old_key]['rmse_m']
        new_v = val['rmse_mean']
        diff  = abs(new_v-old_v)/old_v*100
        max_diff = max(max_diff, diff)
print(f"  MC=10 vs MC=15 최대 차이: {max_diff:.2f}%")
print(f"  → {'안정적' if max_diff < 5 else '차이 큼 — 확인 필요'}")

# ceiling R² 비교
print("\n=== Ceiling R² 비교 (MC=10 vs MC=15) ===")
print(f"{'':20}  MC=10   MC=15   Δ")
for tname,_ in TRAJS:
    for tr in TROLLS:
        tstr=f"{tr:.2f}"
        # MC=10
        sub10={k:v for k,v in ref.items() if k.startswith(f'{tname}_tr{tstr}')}
        Nv=np.array([int(k.split('N')[1]) for k in sub10])
        yv10=np.array([v['rmse_m'] for v in sub10.values()])
        # MC=15
        sub15={k:v for k,v in results.items() if k.startswith(f'{tname}_tr{tstr}')}
        yv15=np.array([results[k]['rmse_mean'] for k in sorted(sub15,key=lambda x:int(x.split('N')[1]))])
        Nv15=np.array(sorted([int(k.split('N')[1]) for k in sub15]))
        try:
            po10,_=curve_fit(cm,Nv,yv10,p0=[min(yv10),0.1])
            r2_10=1-np.sum((yv10-cm(Nv,*po10))**2)/np.sum((yv10-np.mean(yv10))**2)
            po15,_=curve_fit(cm,Nv15,yv15,p0=[min(yv15),0.1])
            r2_15=1-np.sum((yv15-cm(Nv15,*po15))**2)/np.sum((yv15-np.mean(yv15))**2)
            print(f"  {tname} tr={int(tr*100)}%:  {r2_10:.3f}  →  {r2_15:.3f}  Δ={r2_15-r2_10:+.3f}")
        except: pass

elapsed=time.time()-t0
out={
    'config':{'MC':MC,'speed':V,'Ns':NS,'trolls':TROLLS,
               'engine':'paper_fullscale.py',
               'seed_pattern':'seed = i*31 + N',
               'note':'MC=10 → MC=15 upgrade. raw runs stored per condition.'},
    'results':results,
    'metadata':{'elapsed_sec':round(elapsed,1),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
}
with open('E2f_mc15.json','w') as f: json.dump(out,f,indent=2)
print(f"\nSaved: E2f_mc15.json  ({elapsed:.1f}s)")
print("Done.")
