"""
campaign_mechanism.py  —  #12 zigzag 메커니즘 (time-resolved) [campaign.py 동일 엔진/seed]
================================================================
리뷰어 #12: zigzag N-독립을 vote-consensus + response-lag 의 시계열로 설명.

가설(검증 대상, 단정 아님):
  - zigzag: 코너에서 목표 방향이 급반전 → response delay(DELAY_F=26f) 가 추적오차 지배
            → consensus 정확도(R̄, N 의존)와 무관 → N 늘려도 개선 미미 (N-독립)
  - circle: 방향 변화 완만 → lag 영향 작고 consensus 정확도가 지배 → N-benefit

로깅(프레임별): R̄(t), tracking error(t), 목표방향 변화율(turn rate), agent-target 방향오차
슬라이스: {circle, zigzag} × N{5,200} × tr{5,20}% × MC=10 (시계열은 대표 trace)
seed: campaign.make_seed (동일 파이프라인), T0 baseline

출력: mechanism_timeseries.json  (조건별 프레임 시계열 평균 + 요약통계)
실행: python campaign_mechanism.py   (adversary_ladder.py, campaign.py 같은 폴더)
"""
import numpy as np, json, time
import campaign as C
from adversary_ladder import (Circle, Zigzag, DIRS, DT, FRAMES, VOTE_INT,
                              SMOOTH, DELAY_F, LOOK, a2d, _honest_block)

def sim_trace(traj, N, tr, seed, speed=5.0):
    """campaign sim 과 동일 동역학 + 프레임별 시계열 로깅(T0=uniform troll)."""
    rng=np.random.default_rng(seed)
    pos=traj.start(); vel=np.zeros(2); ph=[pos.copy()]
    pang=0.; cur_dir=np.array([1.,0.]); cur_g=0.5
    err=np.empty(FRAMES); Rbar=np.empty(FRAMES); turn=np.empty(FRAMES); head_err=np.empty(FRAMES)
    prev_iang=None
    for f in range(FRAMES):
        if f%VOTE_INT==0:
            di=max(0,len(ph)-1-DELAY_F); dp=ph[di]
            _,arc=traj.closest(dp); lap=traj.at(arc+LOOK)
            idir=lap-dp; n=np.linalg.norm(idir)
            if n>1e-10: idir/=n
            iang=np.degrees(np.arctan2(idir[1],idir[0]))
            # T0 uniform troll: honest + uniform
            honest,nt=_honest_block(iang,pang,tr,N,rng)
            trolls=rng.integers(0,8,nt) if nt>0 else np.array([],dtype=int)
            votes=np.concatenate([honest,trolls]); pang=iang
            bl=DIRS[votes].mean(axis=0); cur_g=np.linalg.norm(bl)
            cur_dir=bl/cur_g if cur_g>1e-10 else np.array([1.,0.])
            # turn rate: 목표 방향 변화량(도)
            if prev_iang is not None:
                dth=abs((iang-prev_iang+180)%360-180)
            else: dth=0.0
            prev_iang=iang; cur_turn=dth
        Rbar[f]=cur_g
        turn[f]=cur_turn if f>=VOTE_INT else 0.0
        # heading error: 현재 속도방향 vs 목표방향
        vd=np.degrees(np.arctan2(vel[1],vel[0])) if np.linalg.norm(vel)>1e-9 else 0.0
        head_err[f]=abs((vd-pang+180)%360-180)
        vel+=SMOOTH*(cur_dir*speed-vel); pos=pos+vel*DT; ph.append(pos.copy())
        cp,_=traj.closest(pos); err[f]=np.linalg.norm(pos-cp)
    return {'rmse':float(np.sqrt(np.mean(err**2))),
            'Rbar_mean':float(Rbar.mean()),'turn_mean':float(turn.mean()),
            'head_err_mean':float(head_err.mean()),
            'err_t':err.tolist(),'Rbar_t':Rbar.tolist(),'turn_t':turn.tolist(),
            'head_err_t':head_err.tolist()}

def run():
    TRAJ={'circle':Circle(),'zigzag':Zigzag()}
    out={}; t0=time.time(); MC=10
    for tname,tobj in TRAJ.items():
        for N in [5,200]:
            for tr in [0.05,0.20]:
                # MC 평균 시계열 + 요약
                accum={k:None for k in ['err_t','Rbar_t','turn_t','head_err_t']}
                summ={'rmse':[],'Rbar_mean':[],'turn_mean':[],'head_err_mean':[]}
                for i in range(MC):
                    seed=C.make_seed(tname,N,tr,'T0',i)
                    r=sim_trace(tobj,N,tr,seed)
                    for k in accum:
                        a=np.array(r[k]); accum[k]=a if accum[k] is None else accum[k]+a
                    for s in summ: summ[s].append(r[s])
                key=f"{tname}_N{N}_tr{tr:.2f}"
                out[key]={**{f'{k}_mean_series':(accum[k]/MC).round(5).tolist() for k in accum},
                          **{s:round(float(np.mean(summ[s])),4) for s in summ},
                          'MC':MC,'N':N,'tr':tr,'traj':tname}
                print(f"  {key}: RMSE={out[key]['rmse']:.3f} R̄={out[key]['Rbar_mean']:.3f} "
                      f"turn={out[key]['turn_mean']:.1f}deg headErr={out[key]['head_err_mean']:.1f}deg [{time.time()-t0:.0f}s]")
    json.dump({'config':{'MC':MC,'slice':'circle/zigzag x N{5,200} x tr{5,20}','model':'T0',
                         'note':'time-resolved R-bar, error, turn-rate, heading-error for #12 mechanism'},
               'results':out},open('mechanism_timeseries.json','w'))
    print(f"\nSaved: mechanism_timeseries.json ({time.time()-t0:.0f}s)")

if __name__=='__main__': run()
