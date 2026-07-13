"""
Lemniscate + Zigzag troll=15% sweep — v2 (seed 수정)

핵심 수정: seed = i*31 + N  (paper_fullscale.py run_condition 원본)
이전 버전 오류: seed = i*31+N+int(tr*100)  ← 원본과 다른 seed

엔진: paper_fullscale.py 완전 동일
  - Lemniscate(a=7, n=800) lookup table
  - Zigzag(amp=5, ns=10, sx=5)
  - gen_votes, sim 함수 동일
  - seed = i * 31 + N

Sanity check: part1_e2f.json, MC=10, seed=i*31+N → diff < 2%
Output: E2f_troll15.json
"""
import numpy as np, json, time
from scipy.optimize import curve_fit

DT=1/60; SMOOTH=0.2; WIN=0.3; DELAY_F=26; DUR=65.0
FRAMES=int(DUR/DT); LOOK=2.0; VOTE_INT=int(WIN/DT)
S2=np.sqrt(2)/2
DIRS=np.array([[1,0],[S2,S2],[0,1],[-S2,S2],[-1,0],[-S2,-S2],[0,-1],[S2,-S2]])
DA=np.degrees(np.arctan2(DIRS[:,1],DIRS[:,0]))%360

def a2d(angles):
    a=angles%360; d=np.abs(DA[None,:]-a[:,None]); d=np.minimum(d,360-d); return np.argmin(d,axis=1)

class Lemniscate:
    def __init__(self,a=7,n=800):
        self.name='lemniscate'
        ts=np.linspace(0,2*np.pi,n,endpoint=False); sn,cs=np.sin(ts),np.cos(ts); d=1+sn**2
        self.pts=np.column_stack([a*cs/d, a*sn*cs/d])
        dl=np.linalg.norm(np.diff(self.pts,axis=0),axis=1)
        self.arcs=np.concatenate([[0],np.cumsum(dl)]); self.circ=self.arcs[-1]
    def closest(self,p):
        d=np.linalg.norm(self.pts-p,axis=1); i=int(np.argmin(d)); return self.pts[i].copy(),self.arcs[i]
    def at(self,arc):
        a=arc%self.circ; i=min(int(np.searchsorted(self.arcs,a)),len(self.pts)-1); return self.pts[i].copy()
    def start(self): return self.pts[0].copy()

class Zigzag:
    def __init__(self,amp=5,ns=10,sx=5):
        self.name='zigzag'; pts=[np.array([0.,0.])]
        for i in range(ns): pts.append(np.array([(i+1)*sx, amp if i%2==0 else 0.]))
        self.c=np.array(pts); self.segs=[(self.c[i],self.c[i+1]) for i in range(ns)]
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
            if arc<=self.cum[i+1]+1e-9: t=(arc-self.cum[i])/self.lens[i]; return a+np.clip(t,0,1)*(b-a)
        return self.c[-1]
    def start(self): return self.c[0].copy()

def gen_votes(iang,pang,tr,N,rng):
    sc=(1-tr)/0.95; na=round(N*0.70*sc); ns=round(N*0.20*sc); nt=round(N*tr); no=max(0,N-na-ns-nt)
    angs=np.empty(na+ns+no); i=0
    angs[i:i+na]=iang+rng.uniform(-3,3,na); i+=na
    diff=iang-pang
    if diff>180: diff-=360
    if diff<-180: diff+=360
    if ns>0: angs[i:i+ns]=pang+diff*(1-rng.uniform(0.2,0.5,ns)); i+=ns
    if no>0: angs[i:i+no]=iang+rng.uniform(-30,30,no); i+=no
    votes=a2d(angs[:i]); trolls=rng.integers(0,8,nt) if nt>0 else np.array([],dtype=int)
    return np.concatenate([votes,trolls])

def sim(traj,N_ag,tr,seed,speed=5.0):
    rng=np.random.default_rng(seed); pos=traj.start(); vel=np.zeros(2); pos_hist=[pos.copy()]
    pang=0.; cur_dir=np.array([1.,0.]); errs=np.empty(FRAMES)
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
# Sanity check — seed = i*31+N, MC=10 (원본과 동일)
# ================================================================
print("=== Sanity check vs part1_e2f.json ===")
with open('part1_e2f.json') as f: ref=json.load(f)['e2f_results']

lem=Lemniscate(); zz=Zigzag()
checks=[
    ('lemniscate',lem,50, 0.20,'lemniscate_tr0.20_N50'),
    ('lemniscate',lem,200,0.40,'lemniscate_tr0.40_N200'),
    ('zigzag',    zz, 50, 0.20,'zigzag_tr0.20_N50'),
    ('zigzag',    zz, 100,0.40,'zigzag_tr0.40_N100'),
]
all_ok=True
for tname,tobj,N,tr,key in checks:
    runs=[sim(tobj,N,tr,seed=i*31+N) for i in range(10)]  # MC=10, seed=i*31+N
    got=round(np.mean(runs),4); exp=ref[key]['rmse_m']
    diff=abs(got-exp)/exp*100
    status='✓' if diff<2.0 else '✗'
    if status=='✗': all_ok=False
    print(f"  {status} {tname} tr={tr} N={N}: got={got:.4f} ref={exp:.4f} diff={diff:.1f}%")
if not all_ok:
    print("  FAIL — 엔진 불일치. 중단."); exit(1)
print("  All PASS\n")

# ================================================================
# troll=15% sweep — seed=i*31+N, MC=15
# ================================================================
MC=15; V=5.0; TROLL=0.15
NS=[5,10,15,20,25,30,40,50,75,100,150,200]
TRAJS=[('lemniscate',lem),('zigzag',zz)]
total=len(NS)*len(TRAJS)*MC

print(f"=== troll=15% sweep | {total} runs | MC={MC} | seed=i*31+N ===")
results={}; t0=time.time(); done=0

for tname,tobj in TRAJS:
    for N in NS:
        rmses=[sim(tobj,N,TROLL,seed=i*31+N) for i in range(MC)]
        done+=MC
        key=f"{tname}_tr{TROLL:.2f}_N{N}"
        results[key]={'rmse_mean':round(float(np.mean(rmses)),4),'rmse_std':round(float(np.std(rmses)),4),
                      'rmse_ci95':round(float(1.96*np.std(rmses)/np.sqrt(MC)),4),'mc_runs':MC}
        eta=(total-done)/(done/(time.time()-t0)) if done>0 else 0
        print(f"  {key}: {results[key]['rmse_mean']:.4f} ± {results[key]['rmse_std']:.4f}  [{done}/{total} ETA {eta:.0f}s]")

def cm(N,a,b): return a+b/np.sqrt(N)
fits={}
print()
print("=== Ceiling R² | troll=10%(기존) vs troll=15%(신규) vs troll=20%(기존) ===")
for tname,_ in TRAJS:
    sub15={k:v for k,v in results.items() if k.startswith(tname)}
    Nv=np.array([int(k.split('N')[1]) for k in sub15]); yv=np.array([v['rmse_mean'] for v in sub15.values()])
    try:
        po,_=curve_fit(cm,Nv,yv,p0=[min(yv),0.1]); r2_15=1-np.sum((yv-cm(Nv,*po))**2)/np.sum((yv-np.mean(yv))**2)
        fits[tname]={'a':round(float(po[0]),4),'b':round(float(po[1]),4),'r2':round(float(r2_15),4)}
    except Exception as e: r2_15=float('nan'); fits[tname]={'error':str(e)}
    r2s={}
    for tstr in ['0.10','0.20']:
        sub={k:v for k,v in ref.items() if k.startswith(f'{tname}_tr{tstr}')}
        Nv2=np.array([int(k.split('N')[1]) for k in sub]); yv2=np.array([v['rmse_m'] for v in sub.values()])
        try:
            po2,_=curve_fit(cm,Nv2,yv2,p0=[min(yv2),0.1]); r2=1-np.sum((yv2-cm(Nv2,*po2))**2)/np.sum((yv2-np.mean(yv2))**2)
            r2s[tstr]=r2
        except: r2s[tstr]=float('nan')
    print(f"  {tname}: tr=10% R²={r2s['0.10']:.3f} | tr=15% R²={r2_15:.3f}★ | tr=20% R²={r2s['0.20']:.3f}")
    if not np.isnan(r2s.get('0.10',float('nan'))): print(f"    jump 10→15: {r2_15-r2s['0.10']:+.3f}")

elapsed=time.time()-t0
out={'config':{'MC':MC,'speed':V,'troll':TROLL,'Ns':NS,
               'engine':'paper_fullscale.py — Lemniscate(a=7,n=800), Zigzag(amp=5,ns=10,sx=5)',
               'seed_pattern':'seed = i*31 + N  (run_condition default)'},
     'results':results,'ceiling_fits':fits,
     'metadata':{'elapsed_sec':round(elapsed,1),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}}
with open('E2f_troll15.json','w') as f: json.dump(out,f,indent=2)
print(f"\nSaved: E2f_troll15.json  ({elapsed:.1f}s)")
print("Done.")
