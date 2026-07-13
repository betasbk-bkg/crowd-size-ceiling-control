"""
campaign.py  —  Paper 2 ("When Crowd Size Stops Mattering") 본 캠페인 [LOCKED v2]
================================================================
PhaseA Freeze v2 동결값:
  궤적   : circle, square, lemniscate, zigzag  (12N 정렬)
  N      : {5,10,15,20,25,30,40,50,75,100,150,200}
  tr     : {5,10,13,15,17,20,25,30,35,40}%      (dense; threshold 부근 조밀)
  모델   : T0 uniform | T1 committed anti-target (c=1) | T2 persistent stale anti-target (c=1, k=10)
  MC     : 50
  속도   : fixed v=5.0 (speed sweep {2,5} 는 별도 supp)
  seed   : SeedSequence([campaign, traj_id, N, tr_milli, model_id, coh_pct, k, mc]) → uint32
           (조건간 충돌 없음, 각 realization 독립 — R1 #7)
  로깅   : rmse, gamma_mean(=R̄), gamma_std, realized_troll_count, seed

  T3/T3o : 본 캠페인 미사용. 게이트 결과를 supp 진단(stabilizer)으로 보고.
  T2 k-sensitivity {5,10,20} : 대표 슬라이스 supp (full grid 아님).

산출: campaign_main_mc50.json  (조건별 mean/std/ci95 + realized count + seed)
실행: python campaign.py            (full 72,000 runs — BK 환경에서)
      python campaign.py --smoke    (파이프라인 검증용 36 runs, 결과해석 안 함)
"""
import numpy as np, json, time, sys, argparse
from adversary_ladder import sim, sim_ext, Circle, Square, Lemniscate, Zigzag

# ---- 동결 격자 ----
TRAJ_BUILD = {'circle':Circle, 'square':Square, 'lemniscate':Lemniscate, 'zigzag':Zigzag}
TRAJ_ID    = {'circle':0, 'square':1, 'lemniscate':2, 'zigzag':3}
NS    = [5,10,15,20,25,30,40,50,75,100,150,200]
TRS   = [0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40]
MODELS= ['T0','T1','T2']
MODEL_ID = {'T0':0,'T1':1,'T2':2}
MC = 50; SPEED = 5.0; K_T2 = 10; CAMPAIGN = 2026

def make_seed(traj, N, tr, model, mc):
    coh = 0 if model=='T0' else 100
    k   = K_T2 if model=='T2' else 0
    ss = np.random.SeedSequence([CAMPAIGN, TRAJ_ID[traj], N, int(round(tr*1000)),
                                 MODEL_ID[model], coh, k, mc])
    return int(ss.generate_state(1)[0])

def run_one(traj_obj, traj_name, N, tr, model, mc):
    seed = make_seed(traj_name, N, tr, model, mc)
    if model == 'T0':
        out = sim(traj_obj, N, tr, seed=seed, model='T1', coherence=0.0, speed=SPEED)   # c=0 == uniform
    elif model == 'T1':
        out = sim(traj_obj, N, tr, seed=seed, model='T1', coherence=1.0, speed=SPEED)
    elif model == 'T2':
        out = sim_ext(traj_obj, N, tr, seed=seed, model='T2', coherence=1.0, speed=SPEED, k_hold=K_T2)
    return out, seed

def manifest():
    total = len(TRAJ_BUILD)*len(NS)*len(TRS)*len(MODELS)*MC
    print(f"[MANIFEST] {len(TRAJ_BUILD)} traj × {len(NS)} N × {len(TRS)} tr × {len(MODELS)} models × MC{MC}")
    print(f"           = {len(TRAJ_BUILD)*len(NS)*len(TRS)*len(MODELS)} conditions × {MC} = {total:,} runs")
    return total

def aggregate(rmses, gammas, realized, seeds):
    a=np.array(rmses)
    return {'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
            'rmse_ci95':round(float(1.96*a.std()/np.sqrt(len(a))),4),
            'gamma_mean':round(float(np.mean(gammas)),4),
            'realized_troll_count':realized,'mc_runs':len(a),
            'seed_first':seeds[0]}

def run(smoke=False):
    if smoke:
        trajs={'circle':Circle()}; ns=[30,100]; trs=[0.20,0.40]; mc=3
        print("=== SMOKE (파이프라인 검증 전용 — 결과 해석 안 함, D5) ===")
    else:
        trajs={n:b() for n,b in TRAJ_BUILD.items()}; ns=NS; trs=TRS; mc=MC
        manifest()
    results={}; t0=time.time(); done=0
    total=len(trajs)*len(ns)*len(trs)*len(MODELS)*mc
    for tname,tobj in trajs.items():
        for N in ns:
            for tr in trs:
                for model in MODELS:
                    rmses=[];gammas=[];seeds=[]
                    for i in range(mc):
                        out,seed=run_one(tobj,tname,N,tr,model,i)
                        rmses.append(out['rmse']);gammas.append(out['gamma_mean']);seeds.append(seed)
                        done+=1
                    key=f"{tname}_tr{tr:.2f}_N{N}_{model}"
                    results[key]=aggregate(rmses,gammas,round(N*tr),seeds)
                    if smoke or done%3000==0:
                        eta=(total-done)/(done/(time.time()-t0)) if done>0 else 0
                        print(f"  {key}: RMSE={results[key]['rmse_mean']:.4f} R̄={results[key]['gamma_mean']:.3f} "
                              f"realized={results[key]['realized_troll_count']} [{done}/{total} ETA {eta:.0f}s]")
    out={'config':{'MC':mc,'speed':SPEED,'Ns':ns,'trolls':trs,'models':MODELS,'k_T2':K_T2,
                   'seed':'SeedSequence([campaign,traj,N,tr_milli,model,coh,k,mc])',
                   'engine':'adversary_ladder.py','smoke':smoke},
         'results':results,
         'metadata':{'elapsed_sec':round(time.time()-t0,1),'total_runs':done,
                     'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}}
    fn='campaign_smoke.json' if smoke else 'campaign_main_mc50.json'
    json.dump(out,open(fn,'w'),indent=2)
    print(f"\nSaved: {fn}  ({done} runs, {time.time()-t0:.0f}s)")
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--manifest',action='store_true')
    a=ap.parse_args()
    if a.manifest: manifest()
    else: run(smoke=a.smoke)
