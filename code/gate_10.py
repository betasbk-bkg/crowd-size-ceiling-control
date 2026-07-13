"""
gate_10.py  —  #10 게이트 (본 캠페인 진입 조건)
================================================================
질문: 추가 adversary(T1 anti-target / T3 aggregate-opposite)가
      uniform(T0)보다 *구별되는* 조작 압력을 만드는가?

predeclared 규칙 (돌리기 전에 고정):
  - 슬라이스: {circle, zigzag} × N{30,100} × tr{0.20,0.40} × {T0,T1,T3} × MC=30
  - 지표: RMSE
  - 비교: 각 조건에서 Δ = mean(RMSE_M) - mean(RMSE_T0), M∈{T1,T3}
          1000-부트스트랩 95% CI. "distinguishable-worse" = CI 하한 > 0 (즉 uniform보다 유의하게 나쁨=강함)
  - GATE GO : T1 또는 T3 가, 8개 슬라이스 조건 중 ≥2개에서 distinguishable-worse,
              그리고 그 중 최소 1개가 tr=40% 조건.
  - GATE STOP: 둘 다 위 기준 미달 → 48k 태우기 전 adversary 재설계.

정의 (PhaseA Freeze v2):
  T1 = reference target-directed direction 의 반대 (open-loop). iang 은 에이전트 상태→lookahead
       reference target point 방향 (공개 가시) — honest 내부모델 아님.
  T3 = 직전 voting window 의 *공개 aggregate* crowd command 반대 (realistic, not oracle).
"""
import numpy as np, time
from adversary_ladder import sim, Circle, Zigzag

print("="*64)
print("#10 GATE — predeclared rule (위 docstring). 결과 미리 안 씀(D5).")
print("="*64)

TRAJ = {'circle': Circle(), 'zigzag': Zigzag()}
NS = [30, 100]; TRS = [0.20, 0.40]; MODELS = ['T0','T1','T3']; MC = 30
rng_boot = np.random.default_rng(20260611)

def runs(traj, N, tr, model):
    # 검증된 seed 패턴; coherence=1 (최대 담합); T3 obs_delay=1
    coh = 0.0 if model=='T0' else 1.0
    return np.array([sim(traj, N, tr, seed=i*31+N+int(tr*100),
                         model=('T1' if model=='T0' else model),  # T0 = T1@c0 (=uniform, 비트동일)
                         coherence=coh, t3_obs_delay=1)['rmse'] for i in range(MC)])

def boot_ci(a, b, it=1000):
    # mean(b) - mean(a) 의 부트스트랩 95% CI
    na, nb = len(a), len(b); d=np.empty(it)
    for k in range(it):
        d[k]=b[rng_boot.integers(0,nb,nb)].mean() - a[rng_boot.integers(0,na,na)].mean()
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

t0=time.time(); rows=[]
print(f"\n{'traj':8} {'N':>4} {'tr':>5} | {'T0':>7} {'T1':>7} {'T3':>7} | T1-T0 [CI]            T3-T0 [CI]")
go_hits = {'T1':[], 'T3':[]}
for tname,tobj in TRAJ.items():
    for N in NS:
        for tr in TRS:
            r0=runs(tobj,N,tr,'T0'); r1=runs(tobj,N,tr,'T1'); r3=runs(tobj,N,tr,'T3')
            lo1,hi1=boot_ci(r0,r1); lo3,hi3=boot_ci(r0,r3)
            d1=r1.mean()-r0.mean(); d3=r3.mean()-r0.mean()
            worse1 = lo1>0; worse3 = lo3>0
            if worse1: go_hits['T1'].append((tname,N,tr))
            if worse3: go_hits['T3'].append((tname,N,tr))
            tag1='WORSE✓' if worse1 else ('better' if hi1<0 else 'flat')
            tag3='WORSE✓' if worse3 else ('better' if hi3<0 else 'flat')
            print(f"{tname:8} {N:>4} {tr:>5.2f} | {r0.mean():7.4f} {r1.mean():7.4f} {r3.mean():7.4f} | "
                  f"{d1:+.4f}[{lo1:+.3f},{hi1:+.3f}]{tag1:>7}  {d3:+.4f}[{lo3:+.3f},{hi3:+.3f}]{tag3:>7}")

def verdict(model):
    hits=go_hits[model]; has_hi=any(tr==0.40 for _,_,tr in hits)
    return len(hits)>=2 and has_hi, hits, has_hi
g1,h1,hi1=verdict('T1'); g3,h3,hi3=verdict('T3')
GO = g1 or g3
print("\n"+"-"*64)
print(f"T1: distinguishable-worse {len(h1)}/8 조건 {h1} | tr40 포함={hi1} | 기준충족={g1}")
print(f"T3: distinguishable-worse {len(h3)}/8 조건 {h3} | tr40 포함={hi3} | 기준충족={g3}")
print(f"\n>>> GATE = {'GO (본 캠페인 진입)' if GO else 'STOP (adversary 재설계)'}  [{time.time()-t0:.0f}s]")
