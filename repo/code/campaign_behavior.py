"""
campaign_behavior.py  —  #9 behavioral sensitivity panel [campaign.py 동일 엔진/seed]
================================================================
리뷰어 #9: 일반 participant model 이 고정적 → learning/fatigue/imitation/coordination/
           variable-delay 가 결론을 바꾸는지 민감도 검증. (troll 강화=#10 와 별개)

목적: 주 결론(N-scaling의 geometry/adversary 조건성)이 *단일 고정 행동모델*에 의존하는지 테스트.
      각 variant 를 켠 채 RMSE 가 baseline(B0) 대비 결론을 뒤집는지 본다 (단정 아님, D5).

행동 variant (honest 그룹에만 적용 — troll 은 모델별 T0/T1/T2 그대로):
  B0 baseline   : 원 모델 (변형 없음)
  B1 var-delay  : 참가자별 반응지연 분산 (DELAY_F → 참가자마다 ±)
  B2 fatigue    : 시간경과에 따라 각도 노이즈 증가 (drift)
  B3 imitation  : 직전 공개 합의(다수추종)로 일부 참가자가 끌림
  B4 learning   : 시간경과에 따라 정확도 향상 (노이즈 감소)

설계 (12,000-run panel):
  circle/square/lemniscate/zigzag × N{30,100} × tr{15,40}% × {T0,T1,T2} × {B0..B4} × MC50
  = 4 × 2 × 2 × 3 × 5 × 50 = 12,000 runs

seed: campaign.make_seed 확장 (behavior id 추가) — 충돌 없음
출력: behavior_panel.json
실행: python campaign_behavior.py   (adversary_ladder.py, campaign.py 같은 폴더)
"""
import numpy as np, json, time
import campaign as C
from adversary_ladder import (Circle, Square, Lemniscate, Zigzag, DIRS, DT, FRAMES,
                              VOTE_INT, SMOOTH, DELAY_F, LOOK, a2d, _anti_idx)

TRAJ_B={'circle':Circle,'square':Square,'lemniscate':Lemniscate,'zigzag':Zigzag}
BEHAV=['B0','B1','B2','B3','B4']
BEH_ID={b:i for i,b in enumerate(BEHAV)}

def gen_votes_behavior(iang, pang, tr, N, rng, model, behavior, frame,
                       prev_pub_deg, troll_hold):
    """honest 그룹에 behavior variant 적용 + troll 은 model(T0/T1/T2)."""
    sc=(1-tr)/0.95
    na=round(N*0.70*sc); ns=round(N*0.20*sc); nt=round(N*tr); no=max(0,N-na-ns-nt)
    prog=frame/FRAMES  # 0→1 시간 진행
    # 기본 정확 그룹 노이즈 폭
    acc_noise=3.0
    if behavior=='B2':   acc_noise=3.0+9.0*prog          # fatigue: 3°→12°
    elif behavior=='B4': acc_noise=8.0-5.0*prog          # learning: 8°→3°
    angs=np.empty(na+ns+no); i=0
    # 정확 그룹
    base=iang+rng.uniform(-acc_noise,acc_noise,na)
    if behavior=='B3' and prev_pub_deg is not None:       # imitation: 30% 가 공개합의로 끌림
        k=int(na*0.30)
        if k>0: base[:k]=0.7*base[:k]+0.3*prev_pub_deg
    angs[i:i+na]=base; i+=na
    # 지연 그룹
    diff=iang-pang
    if diff>180: diff-=360
    if diff<-180: diff+=360
    if ns>0:
        lag_frac=rng.uniform(0.2,0.5,ns)
        if behavior=='B1':                                # var-delay: 지연 분산 확대
            lag_frac=np.clip(rng.uniform(0.1,0.7,ns),0,1)
        angs[i:i+ns]=pang+diff*(1-lag_frac); i+=ns
    # 기타(노이즈) 그룹
    if no>0: angs[i:i+no]=iang+rng.uniform(-30,30,no); i+=no
    honest=a2d(angs[:i])
    # troll
    if nt>0:
        if model=='T0':
            trolls=rng.integers(0,8,nt)
        elif model=='T1':
            trolls=np.full(nt,_anti_idx(iang),dtype=int)
        elif model=='T2':
            if troll_hold['left']<=0:
                troll_hold['idx']=_anti_idx(iang); troll_hold['left']=troll_hold['k']
            troll_hold['left']-=1; trolls=np.full(nt,troll_hold['idx'],dtype=int)
    else:
        trolls=np.array([],dtype=int)
    return np.concatenate([honest,trolls]), troll_hold

def sim_behavior(traj,N,tr,seed,model,behavior,speed=5.0,k_hold=10):
    rng=np.random.default_rng(seed); pos=traj.start(); vel=np.zeros(2); ph=[pos.copy()]
    pang=0.; cur_dir=np.array([1.,0.]); cur_g=0.5; errs=np.empty(FRAMES); gammas=np.empty(FRAMES)
    pub_hist=[]; hold={'idx':None,'left':0,'k':k_hold}
    for f in range(FRAMES):
        if f%VOTE_INT==0:
            di=max(0,len(ph)-1-DELAY_F); dp=ph[di]
            _,arc=traj.closest(dp); lap=traj.at(arc+LOOK); idir=lap-dp; n=np.linalg.norm(idir)
            if n>1e-10: idir/=n
            iang=np.degrees(np.arctan2(idir[1],idir[0]))
            prev_pub=pub_hist[-1] if pub_hist else None
            votes,hold=gen_votes_behavior(iang,pang,tr,N,rng,model,behavior,f,prev_pub,hold); pang=iang
            bl=DIRS[votes].mean(axis=0); cur_g=np.linalg.norm(bl)
            cur_dir=bl/cur_g if cur_g>1e-10 else np.array([1.,0.])
            pub_hist.append(np.degrees(np.arctan2(cur_dir[1],cur_dir[0])))
        gammas[f]=cur_g
        vel+=SMOOTH*(cur_dir*speed-vel); pos=pos+vel*DT; ph.append(pos.copy())
        cp,_=traj.closest(pos); errs[f]=np.linalg.norm(pos-cp)
    return {'rmse':float(np.sqrt(np.mean(errs**2))),'gamma_mean':float(gammas.mean())}

def make_seed_b(traj,N,tr,model,behavior,mc):
    ss=np.random.SeedSequence([C.CAMPAIGN,C.TRAJ_ID[traj],N,int(round(tr*1000)),
                               C.MODEL_ID[model],100 if model!='T0' else 0,
                               C.K_T2 if model=='T2' else 0, BEH_ID[behavior]*1000+mc])
    return int(ss.generate_state(1)[0])

def run():
    trajs={n:b() for n,b in TRAJ_B.items()}
    NS=[30,100]; TRS=[0.15,0.40]; MODELS=['T0','T1','T2']; MC=50
    total=len(trajs)*len(NS)*len(TRS)*len(MODELS)*len(BEHAV)*MC
    print(f"#9 behavioral panel: {total} runs (4traj×2N×2tr×3model×5behav×MC{MC})")
    res={}; t0=time.time(); done=0
    for tn,to in trajs.items():
        for N in NS:
            for tr in TRS:
                for model in MODELS:
                    for beh in BEHAV:
                        rm=[]
                        for i in range(MC):
                            seed=make_seed_b(tn,N,tr,model,beh,i)
                            rm.append(sim_behavior(to,N,tr,seed,model,beh)['rmse'])
                            done+=1
                        a=np.array(rm)
                        res[f"{tn}_N{N}_tr{tr:.2f}_{model}_{beh}"]={
                            'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
                            'rmse_ci95':round(float(1.96*a.std()/np.sqrt(MC)),4),'mc_runs':MC}
                    if done%1500==0:
                        print(f"  [{done}/{total}] {tn} N{N} tr{int(tr*100)} {model} done [{time.time()-t0:.0f}s]")
    json.dump({'config':{'MC':MC,'Ns':NS,'trolls':TRS,'models':MODELS,'behaviors':BEHAV,
                         'behavior_defs':{'B0':'baseline','B1':'variable delay variance',
                          'B2':'fatigue-like noise growth 3->12deg','B3':'imitation 30% follow public consensus',
                          'B4':'learning-like noise decay 8->3deg'},
                         'note':'#9 behavioral sensitivity — variants on honest group'},
               'results':res,'metadata':{'total_runs':done,'elapsed_sec':round(time.time()-t0,1)}},
              open('behavior_panel.json','w'),indent=2)
    print(f"\nSaved: behavior_panel.json ({done} runs, {time.time()-t0:.0f}s)")

if __name__=='__main__': run()
