"""
adversary_ladder.py  —  Paper 2 (Access-2026-17026) #10 응답
================================================================
사전명세 adversary ladder T0–T3 + coherence c.
검증된 엔진(run_E2_E3_proper.py / paper_fullscale.py)을 그대로 내장하고,
gen_votes 의 **troll 생성 블록만** 교체한다 (honest 그룹은 불변).

LOCK (PhaseA_Freeze §3,§4):
  T0  uniform random                      θ ~ U(8-dir)
  T1  committed anti-target (open-loop)   θ = quantize8(iang + 180)   [주]
  T3  adaptive consensus-opposite         θ = quantize8(prev_consensus + 180), obs delay [보조]
  coherence c : committed 중 floor(c*nt) 명만 집중(anti), 나머지 uniform
                c=0  -> uniform 재현(원본과 비트 일치),  c=1 -> 완전담합

  T2 (persistent/burst) 는 보조 — 파라미터(k/burst) 추후 매니페스트에서 확정. 하단 stub.

seed = i*31 + N + int(tr*100)   (circle/square, run_E2_E3_proper 동일)
fixed velocity, MC=50 (신규).  γ 미사용 — R̄ 는 진단 로깅만.
"""
import numpy as np

# ===== 엔진 상수 (검증분, PhaseA_Freeze §2) =====
DT = 1/60; MSPD = 5.0; SMOOTH = 0.2; WIN = 0.3; DELAY_F = 26
DUR = 65.0; FRAMES = int(DUR/DT); LOOK = 2.0; VOTE_INT = int(WIN/DT)  # 3900, 18
S2 = np.sqrt(2)/2
DIRS = np.array([[1,0],[S2,S2],[0,1],[-S2,S2],[-1,0],[-S2,-S2],[0,-1],[S2,-S2]])
DA = np.degrees(np.arctan2(DIRS[:,1], DIRS[:,0])) % 360

def a2d(angles):
    a = angles % 360
    d = np.abs(DA[None,:] - a[:,None]); d = np.minimum(d, 360-d)
    return np.argmin(d, axis=1)

def _anti_idx(angle_deg):
    """타깃/합의 방향(deg)의 8-dir 반대 인덱스"""
    return int(a2d(np.array([angle_deg + 180.0]))[0])

# ===== 궤적 (검증분) =====
class Circle:
    def __init__(s, R=10): s.R = R
    def closest(s, p):
        t = np.arctan2(p[1], p[0]); return s.R*np.array([np.cos(t),np.sin(t)]), (t%(2*np.pi))*s.R
    def at(s, arc): t = arc/s.R; return s.R*np.array([np.cos(t), np.sin(t)])
    def start(s): return np.array([s.R, 0.])

class Square:
    def __init__(s, h=10):
        s.c = np.array([[h,0],[h,h],[-h,h],[-h,-h],[h,-h],[h,0.]], dtype=float)
        s.segs = [(s.c[i], s.c[i+1]) for i in range(5)]
        s.lens = [np.linalg.norm(b-a) for a,b in s.segs]; s.circ = sum(s.lens)
        s.cum = np.array([0]+list(np.cumsum(s.lens)))
    def closest(s, p):
        bd, bp, ba = 1e10, s.c[0], 0.
        for i,(a,b) in enumerate(s.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2,0,1); pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,s.cum[i]+t*s.lens[i]
        return bp, ba
    def at(s, arc):
        arc=arc%s.circ
        for i,(a,b) in enumerate(s.segs):
            if arc<=s.cum[i+1]+1e-9:
                t=(arc-s.cum[i])/s.lens[i]; return a+np.clip(t,0,1)*(b-a)
        return s.c[-1]
    def start(s): return s.c[0].copy()

class Lemniscate:
    def __init__(self, a=7, n=800):
        ts=np.linspace(0,2*np.pi,n,endpoint=False); sn,cs=np.sin(ts),np.cos(ts); den=1+sn**2
        self.pts=np.column_stack([a*cs/den, a*sn*cs/den])
        dl=np.linalg.norm(np.diff(self.pts,axis=0),axis=1)
        self.arcs=np.concatenate([[0],np.cumsum(dl)]); self.circ=self.arcs[-1]
    def closest(self, p):
        d=np.linalg.norm(self.pts-p,axis=1); i=int(np.argmin(d)); return self.pts[i].copy(), self.arcs[i]
    def at(self, arc):
        a=arc%self.circ; i=min(int(np.searchsorted(self.arcs,a)),len(self.pts)-1); return self.pts[i].copy()
    def start(self): return self.pts[0].copy()

class Zigzag:
    def __init__(self, amp=5, ns=10, sx=5):
        pts=[np.array([0.,0.])]
        for i in range(ns): pts.append(np.array([(i+1)*sx, amp if i%2==0 else 0.]))
        self.c=np.array(pts); self.segs=[(self.c[i],self.c[i+1]) for i in range(ns)]
        self.lens=[np.linalg.norm(b-a) for a,b in self.segs]
        self.circ=sum(self.lens); self.cum=np.array([0]+list(np.cumsum(self.lens)))
    def closest(self, p):
        bd,bp,ba=1e10,self.c[0],0.
        for i,(a,b) in enumerate(self.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2,0,1); pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba
    def at(self, arc):
        arc=arc%self.circ
        for i,(a,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9: t=(arc-self.cum[i])/self.lens[i]; return a+np.clip(t,0,1)*(b-a)
        return self.c[-1]
    def start(self): return self.c[0].copy()

TRAJ = {'circle':Circle, 'square':Square, 'lemniscate':Lemniscate, 'zigzag':Zigzag}

# ===== honest 그룹 (원본과 100% 동일) =====
def _honest_block(iang, pang, tr, N, rng):
    sc=(1-tr)/0.95; na=round(N*0.70*sc); ns=round(N*0.20*sc); nt=round(N*tr); no=max(0,N-na-ns-nt)
    angs=np.empty(na+ns+no); i=0
    angs[i:i+na]=iang+rng.uniform(-3,3,na); i+=na
    diff=iang-pang
    if diff>180: diff-=360
    if diff<-180: diff+=360
    if ns>0: angs[i:i+ns]=pang+diff*(1-rng.uniform(0.2,0.5,ns)); i+=ns
    if no>0: angs[i:i+no]=iang+rng.uniform(-30,30,no); i+=no
    return a2d(angs[:i]), nt

# ===== adversary 블록 (교체 지점) =====
def _troll_block(model, nt, coherence, iang, prev_consensus_deg, rng):
    """nt 개 troll 의 8-dir 투표 인덱스 반환. c=0 -> 원본 uniform 과 비트 동일."""
    if nt <= 0:
        return np.array([], dtype=int)
    if model == 'T0' or coherence <= 0.0:
        return rng.integers(0, 8, nt)                       # 원본과 동일
    n_conc = int(np.floor(coherence * nt))
    n_disp = nt - n_conc
    if model == 'T1':
        anti = _anti_idx(iang)                              # 타깃 반대 (open-loop)
    elif model == 'T3':
        base = iang if prev_consensus_deg is None else prev_consensus_deg
        anti = _anti_idx(base)                              # 합의 반대 (closed-loop, 지연관측)
    else:
        raise NotImplementedError(f"model {model} (T2 는 stub — 하단 참조)")
    conc = np.full(n_conc, anti, dtype=int)
    disp = rng.integers(0, 8, n_disp) if n_disp > 0 else np.array([], dtype=int)
    return np.concatenate([conc, disp])

def gen_votes_adv(iang, pang, tr, N, rng, model='T0', coherence=1.0, prev_consensus_deg=None):
    honest, nt = _honest_block(iang, pang, tr, N, rng)
    trolls = _troll_block(model, nt, coherence, iang, prev_consensus_deg, rng)
    return np.concatenate([honest, trolls])

# ===== 시뮬 (fixed velocity; R̄ 로깅; T3용 합의 추적) =====
def sim(traj, N, tr, seed, model='T0', coherence=1.0, speed=5.0, t3_obs_delay=1, log_series=False):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); pos_hist=[pos.copy()]
    pang=0.; cur_dir=np.array([1.,0.]); cur_gamma=0.5
    errs=np.empty(FRAMES); gammas=np.empty(FRAMES)
    consensus_hist=[]   # 8-dir deg of honest+troll aggregate per vote-interval (T3 관측원)
    series=[] if log_series else None
    for f in range(FRAMES):
        if f % VOTE_INT == 0:
            di=max(0,len(pos_hist)-1-DELAY_F); dp=pos_hist[di]
            _,arc=traj.closest(dp); lap=traj.at(arc+LOOK)
            idir=lap-dp; n=np.linalg.norm(idir)
            if n>1e-10: idir/=n
            iang=np.degrees(np.arctan2(idir[1],idir[0]))
            # T3: 지연관측한 직전 합의방향(deg)
            prev_deg=None
            if model=='T3' and len(consensus_hist)>=t3_obs_delay:
                prev_deg=consensus_hist[-t3_obs_delay]
            votes=gen_votes_adv(iang,pang,tr,N,rng,model,coherence,prev_deg); pang=iang
            bl=DIRS[votes].mean(axis=0); cur_gamma=np.linalg.norm(bl)
            cur_dir=bl/cur_gamma if cur_gamma>1e-10 else np.array([1.,0.])
            consensus_hist.append(np.degrees(np.arctan2(cur_dir[1],cur_dir[0])))
            if log_series:
                series.append((f, float(cur_gamma)))
        gammas[f]=cur_gamma
        vel += SMOOTH*(cur_dir*speed - vel)          # fixed velocity
        pos = pos + vel*DT; pos_hist.append(pos.copy())
        cp,_=traj.closest(pos); errs[f]=np.linalg.norm(pos-cp)
    out={'rmse':float(np.sqrt(np.mean(errs**2))),
         'gamma_mean':float(np.mean(gammas)), 'gamma_std':float(np.std(gammas))}  # gamma_mean == R̄ 평균
    if log_series: out['Rbar_series']=series
    return out

# ===== T2 stub (보조 — 파라미터 추후 LOCK) =====
def _troll_block_T2(*a, **k):
    raise NotImplementedError(
        "T2 persistent/burst: per-troll hold-k 또는 burst-window 필요. "
        "troll_state 배열을 sim 루프에 추가해 구현 — B1+ ladder 단계에서 k/burst 확정 후.")

if __name__ == '__main__':
    print("adversary_ladder.py — import용 모듈. 검증은 verify_ladder.py 참조.")

# ===== T2 persistent + T3-oracle (게이트 확장) =====
def gen_votes_ext(iang, pang, tr, N, rng, model, coherence, prev_consensus_deg,
                  prev_honest_deg, troll_hold):
    """T2(persistent), T3o(oracle: 비적대 합의만 관측, 자기표 제외)"""
    honest, nt = _honest_block(iang, pang, tr, N, rng)
    if nt <= 0:
        return np.concatenate([honest, np.array([],dtype=int)]), troll_hold
    n_conc = int(np.floor(coherence*nt)); n_disp = nt-n_conc
    if model == 'T2':   # persistent: 한 번 뽑은 오방향을 hold 동안 유지
        if troll_hold['idx'] is None or troll_hold['left'] <= 0:
            troll_hold['idx'] = _anti_idx(iang)     # 오방향 = 타깃 반대 (고정 후 유지)
            troll_hold['left'] = troll_hold['k']
        troll_hold['left'] -= 1
        anti = troll_hold['idx']
    elif model == 'T3o':  # oracle: 비적대 합의 반대 (자기표 제외)
        base = iang if prev_honest_deg is None else prev_honest_deg
        anti = _anti_idx(base)
    else:
        raise NotImplementedError(model)
    conc = np.full(n_conc, anti, dtype=int)
    disp = rng.integers(0,8,n_disp) if n_disp>0 else np.array([],dtype=int)
    return np.concatenate([honest, conc, disp]), troll_hold

def sim_ext(traj, N, tr, seed, model, coherence=1.0, speed=5.0, t3_obs_delay=1, k_hold=10):
    rng=np.random.default_rng(seed); pos=traj.start(); vel=np.zeros(2); pos_hist=[pos.copy()]
    pang=0.; cur_dir=np.array([1.,0.]); cur_gamma=0.5
    errs=np.empty(FRAMES); gammas_ext=np.empty(FRAMES)
    honest_hist=[]   # 비적대(honest) 부분합의 deg — T3o 관측원
    hold={'idx':None,'left':0,'k':k_hold}
    for f in range(FRAMES):
        if f%VOTE_INT==0:
            di=max(0,len(pos_hist)-1-DELAY_F); dp=pos_hist[di]
            _,arc=traj.closest(dp); lap=traj.at(arc+LOOK)
            idir=lap-dp; n=np.linalg.norm(idir)
            if n>1e-10: idir/=n
            iang=np.degrees(np.arctan2(idir[1],idir[0]))
            prev_h = honest_hist[-t3_obs_delay] if (model=='T3o' and len(honest_hist)>=t3_obs_delay) else None
            votes,hold=gen_votes_ext(iang,pang,tr,N,rng,model,coherence,None,prev_h,hold); pang=iang
            # 전체 합의
            bl=DIRS[votes].mean(axis=0); cur_gamma=np.linalg.norm(bl)
            cur_dir=bl/cur_gamma if cur_gamma>1e-10 else np.array([1.,0.])
            # honest 부분합의(비적대만) 기록 — oracle 관측용
            h,nt=_honest_block(iang,pang,tr,N,np.random.default_rng(seed*7+f))  # 동일분포 재현용 별도 draw
            # 주: 관측원은 '직전' honest 합의 방향. 근사로 idir 사용시 단순화 가능하나 여기선 honest 투표 평균.
            hv=DIRS[h].mean(axis=0); hh=np.degrees(np.arctan2(hv[1],hv[0])) if np.linalg.norm(hv)>1e-10 else iang
            honest_hist.append(hh)
        vel+=SMOOTH*(cur_dir*speed-vel); pos=pos+vel*DT; pos_hist.append(pos.copy())
        gammas_ext[f]=cur_gamma
        cp,_=traj.closest(pos); errs[f]=np.linalg.norm(pos-cp)
    return {'rmse':float(np.sqrt(np.mean(errs**2))),
            'gamma_mean':float(np.mean(gammas_ext)),'gamma_std':float(np.std(gammas_ext))}
