"""
campaign_anova_raw.py  —  #5 ANOVA 가정진단용 raw 슬라이스 [campaign.py 동일 엔진/seed]
================================================================
목적: campaign_main_mc50.json 은 mean/std만 저장 → 잔차 정규성/등분산/robust ANOVA 불가.
      동일 엔진·동일 seed 방식으로 raw MC=50 runs 를 저장하는 슬라이스 재실행.

설계 (원고 Table III ANOVA 정합):
  circle, square × tr {5,15,40}% × 12N × T0(baseline) × MC=50
  = 2 × 3 × 12 × 50 = 3,600 runs

seed: campaign.py 와 동일 SeedSequence (model='T0' 동일 좌표) → main 캠페인과 정합/재현
출력: anova_raw_mc50.json  (rmse_raw 포함)

실행: python campaign_anova_raw.py   (adversary_ladder.py 같은 폴더 필요)
"""
import numpy as np, json, time
from adversary_ladder import sim, Circle, Square
import campaign as C   # make_seed 재사용 — main 캠페인과 동일 seed 보장

TRAJ={'circle':Circle(),'square':Square()}
NS=[5,10,15,20,25,30,40,50,75,100,150,200]
TRS=[0.05,0.15,0.40]   # ANOVA 3-level (low / transition / high)
MC=50; MODEL='T0'

def run():
    results={}; t0=time.time(); done=0
    total=len(TRAJ)*len(NS)*len(TRS)*MC
    print(f"ANOVA raw 슬라이스: {total} runs (circle/square × {len(TRS)}tr × {len(NS)}N × MC{MC}, T0)")
    for tname,tobj in TRAJ.items():
        for tr in TRS:
            for N in NS:
                rmses=[]; seeds=[]
                for i in range(MC):
                    seed=C.make_seed(tname,N,tr,MODEL,i)   # main 캠페인과 동일 seed
                    out=sim(tobj,N,tr,seed=seed,model='T1',coherence=0.0,speed=5.0)  # T0=T1@c0
                    rmses.append(round(out['rmse'],6)); seeds.append(seed)
                    done+=1
                key=f"{tname}_tr{tr:.2f}_N{N}_T0"
                a=np.array(rmses)
                results[key]={'rmse_mean':round(float(a.mean()),4),'rmse_std':round(float(a.std()),4),
                              'rmse_ci95':round(float(1.96*a.std()/np.sqrt(MC)),4),
                              'rmse_raw':rmses,'realized_troll_count':round(N*tr),
                              'mc_runs':MC,'seed_first':seeds[0]}
                if done % 600==0:
                    print(f"  {key}: {a.mean():.4f} [{done}/{total} {time.time()-t0:.0f}s]")
    out={'config':{'MC':MC,'speed':5.0,'Ns':NS,'trolls':TRS,'model':MODEL,
                   'seed':'SeedSequence (campaign.py make_seed, model=T0)',
                   'purpose':'#5 ANOVA assumption diagnostics — raw runs'},
         'results':results,
         'metadata':{'total_runs':done,'elapsed_sec':round(time.time()-t0,1),
                     'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}}
    json.dump(out,open('anova_raw_mc50.json','w'),indent=2)
    print(f"\nSaved: anova_raw_mc50.json ({done} runs, {time.time()-t0:.0f}s)")

if __name__=='__main__':
    run()
