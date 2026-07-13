"""
campaign_speed_mech.py  —  #13 speed 확장 + #12 MC50 확정 [campaign.py 동일 엔진/seed]
================================================================
#13: speed independence 가 circle 만 테스트됨 → lem/zig 의 v=2.0 데이터 생성하여
     전 geometry 확장. (circle/square v2.0 는 기존 E3_v2_sweep.json 재사용)
#12: mechanism 시계열이 MC=10 → 동일 슬라이스 MC=50 으로 RMSE 확정.

두 슬라이스를 한 스크립트로:
  [A] #13 speed:  lemniscate/zigzag × 12N × tr{5,20,40}% × v=2.0 × T0 × MC50  = 2*12*3*50 = 3,600
  [B] #12 MC50 :  circle/zigzag × N{5,200} × tr{5,20}% × v=5.0 × T0 × MC50    = 2*2*2*50  = 400

seed: campaign.make_seed (속도는 좌표에 없으므로 speed_tag 로 분기 — 충돌 방지)
출력: speed_ext.json, mech_mc50.json
실행: python campaign_speed_mech.py
"""
import numpy as np, json, time
import campaign as C
from adversary_ladder import sim, Circle, Square, Lemniscate, Zigzag

def seed_speed(traj,N,tr,mc,v):
    ss=np.random.SeedSequence([C.CAMPAIGN,C.TRAJ_ID[traj],N,int(round(tr*1000)),
                               0,0,0, int(v*10)*100000+mc])  # speed 태그로 분리
    return int(ss.generate_state(1)[0])

def run_speed():
    TRAJ={'lemniscate':Lemniscate(),'zigzag':Zigzag()}
    NS=[5,10,15,20,25,30,40,50,75,100,150,200]; TRS=[0.05,0.20,0.40]; MC=50; V=2.0
    total=len(TRAJ)*len(NS)*len(TRS)*MC
    print(f"[A] #13 speed (lem/zig v=2.0): {total} runs")
    res={}; t0=time.time(); done=0
    for tn,to in TRAJ.items():
        for tr in TRS:
            for N in NS:
                rm=[sim(to,N,tr,seed=seed_speed(tn,N,tr,i,V),model='T1',coherence=0.0,speed=V)['rmse'] for i in range(MC)]
                a=np.array(rm); res[f"{tn}_tr{tr:.2f}_N{N}_v2"]={
                    'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
                    'rmse_ci95':round(float(1.96*a.std()/np.sqrt(MC)),4),'mc_runs':MC}
                done+=MC
        print(f"   {tn} done [{time.time()-t0:.0f}s]")
    json.dump({'config':{'MC':MC,'speed':V,'Ns':NS,'trolls':TRS,'trajs':list(TRAJ),
                         'purpose':'#13 speed-independence extension to lemniscate/zigzag'},
               'results':res},open('speed_ext.json','w'),indent=2)
    print(f"   Saved speed_ext.json ({done} runs)\n")

def run_mech_mc50():
    TRAJ={'circle':Circle(),'zigzag':Zigzag()}; MC=50
    print(f"[B] #12 MC50 confirm: {2*2*2*MC} runs")
    res={}; t0=time.time()
    for tn,to in TRAJ.items():
        for N in [5,200]:
            for tr in [0.05,0.20]:
                rm=[sim(to,N,tr,seed=C.make_seed(tn,N,tr,'T0',i),model='T1',coherence=0.0,speed=5.0) for i in range(MC)]
                # gamma 도 평균
                gm=[x['gamma_mean'] for x in rm]; rr=[x['rmse'] for x in rm]
                a=np.array(rr); res[f"{tn}_N{N}_tr{tr:.2f}"]={
                    'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
                    'rmse_ci95':round(float(1.96*a.std()/np.sqrt(MC)),4),
                    'gamma_mean':round(float(np.mean(gm)),4),'mc_runs':MC}
    json.dump({'config':{'MC':MC,'purpose':'#12 mechanism RMSE confirmation at MC=50'},
               'results':res},open('mech_mc50.json','w'),indent=2)
    print(f"   Saved mech_mc50.json ({time.time()-t0:.0f}s)")

if __name__=='__main__':
    run_speed(); run_mech_mc50()
