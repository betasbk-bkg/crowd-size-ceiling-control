"""
troll=15% sweep — 원본 엔진 (run_E2_E3_proper.py) 그대로 사용
Circle + Square 모두 재실행
"""
import numpy as np, time, json
from scipy.optimize import curve_fit

# ================================================================
# 원본 엔진 (run_E2_E3_proper.py 완전 동일)
# ================================================================
DT = 1/60; MSPD = 5.0; SMOOTH = 0.2; WIN = 0.3; DELAY_F = 26
DUR = 65.0; FRAMES = int(DUR / DT); LOOK = 2.0; VOTE_INT = int(WIN / DT)
S2 = np.sqrt(2) / 2
DIRS = np.array([[1,0],[S2,S2],[0,1],[-S2,S2],[-1,0],[-S2,-S2],[0,-1],[S2,-S2]])
DA = np.degrees(np.arctan2(DIRS[:,1], DIRS[:,0])) % 360

def a2d(angles):
    a = angles % 360
    diffs = np.abs(DA[None,:] - a[:,None])
    diffs = np.minimum(diffs, 360 - diffs)
    return np.argmin(diffs, axis=1)

class Circle:
    def __init__(s, R=10): s.R = R; s.circ = 2*np.pi*R
    def closest(s, p):
        t = np.arctan2(p[1], p[0])
        cp = s.R * np.array([np.cos(t), np.sin(t)])
        return cp, (t % (2*np.pi)) * s.R
    def at(s, arc):
        t = arc / s.R
        return s.R * np.array([np.cos(t), np.sin(t)])
    def start(s): return np.array([s.R, 0.])

class Square:
    def __init__(s, h=10):
        s.c = np.array([[h,0],[h,h],[-h,h],[-h,-h],[h,-h],[h,0.]], dtype=float)
        s.segs = [(s.c[i], s.c[i+1]) for i in range(5)]
        s.lens = [np.linalg.norm(b-a) for a,b in s.segs]
        s.circ = sum(s.lens)
        s.cum = np.array([0] + list(np.cumsum(s.lens)))
    def closest(s, p):
        bd, bp, ba = 1e10, s.c[0], 0.
        for i,(a,b) in enumerate(s.segs):
            v = b - a; l2 = v @ v
            if l2 < 1e-10: continue
            t = np.clip((p-a) @ v / l2, 0, 1)
            pt = a + t*v; d = np.linalg.norm(p - pt)
            if d < bd: bd, bp, ba = d, pt, s.cum[i] + t*s.lens[i]
        return bp, ba
    def at(s, arc):
        arc = arc % s.circ
        for i,(a,b) in enumerate(s.segs):
            if arc <= s.cum[i+1] + 1e-9:
                t = (arc - s.cum[i]) / s.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return s.c[-1]
    def start(s): return s.c[0].copy()

def gen_votes(iang, pang, tr, N, rng):
    sc = (1 - tr) / 0.95
    na = round(N * 0.70 * sc); ns = round(N * 0.20 * sc)
    nt = round(N * tr); no = max(0, N - na - ns - nt)
    angs = np.empty(na + ns + no); i = 0
    angs[i:i+na] = iang + rng.uniform(-3, 3, na); i += na
    diff = iang - pang
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if ns > 0: angs[i:i+ns] = pang + diff*(1-rng.uniform(0.2,0.5,ns)); i += ns
    if no > 0: angs[i:i+no] = iang + rng.uniform(-30, 30, no); i += no
    votes = a2d(angs[:i])
    trolls = rng.integers(0, 8, nt) if nt > 0 else np.array([], dtype=int)
    return np.concatenate([votes, trolls])

def sim(traj, N_agents, tr, seed, speed_override=None):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2); pos_hist = [pos.copy()]
    pang = 0.; cur_dir = np.array([1., 0.]); cur_gamma = 0.5
    V = speed_override if speed_override is not None else MSPD
    errs = np.empty(FRAMES)
    for f in range(FRAMES):
        if f % VOTE_INT == 0:
            di = max(0, len(pos_hist) - 1 - DELAY_F)
            dp = pos_hist[di]
            _, arc = traj.closest(dp)
            lap = traj.at(arc + LOOK)
            idir = lap - dp; n = np.linalg.norm(idir)
            if n > 1e-10: idir /= n
            iang = np.degrees(np.arctan2(idir[1], idir[0]))
            votes = gen_votes(iang, pang, tr, N_agents, rng); pang = iang
            vecs = DIRS[votes]; bl = vecs.mean(axis=0)
            cur_gamma = np.linalg.norm(bl)
            cur_dir = bl / cur_gamma if cur_gamma > 1e-10 else np.array([1., 0.])
        vel += SMOOTH * (cur_dir * V - vel)
        pos = pos + vel * DT; pos_hist.append(pos.copy())
        cp, _ = traj.closest(pos); errs[f] = np.linalg.norm(pos - cp)
    return float(np.sqrt(np.mean(errs**2)))

# ================================================================
# Sanity check vs E3_supplement_proper.json
# ================================================================
print("=== Sanity check (원본 데이터와 비교) ===")
with open('E3_supplement_proper.json') as f:
    e3ref = json.load(f)

c = Circle(); sq = Square()
checks = [
    ('circle', c,  50, 0.20, 'circle_tr0.20_N50',  0.8119),
    ('circle', c,  5,  0.40, 'circle_tr0.40_N5',   1.1424),
    ('square', sq, 50, 0.20, 'square_tr0.20_N50',  1.1576),
    ('square', sq, 5,  0.05, 'square_tr0.05_N5',   1.1624),
]
all_ok = True
for label, traj, N, tr, key, expect in checks:
    runs = [sim(traj, N, tr, seed=i*31+N+int(tr*100), speed_override=5.0) for i in range(15)]
    got = np.mean(runs)
    ref = e3ref['results'][key]['rmse_mean']
    diff_pct = abs(got - ref) / ref * 100
    status = '✓' if diff_pct < 5 else '✗'
    if status == '✗': all_ok = False
    print(f"  {status} {label} tr={tr} N={N}: got={got:.4f} ref={ref:.4f} diff={diff_pct:.1f}%")

print(f"  {'All PASS' if all_ok else 'FAIL — check engine'}\n")
if not all_ok:
    print("  ⚠️  Sanity check 실패. 실험 중단.")
    exit(1)

# ================================================================
# troll=15% sweep
# ================================================================
MC = 15; V = 5.0; TROLL = 0.15
NS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
TRAJS = [('circle', c), ('square', sq)]
total = len(NS) * len(TRAJS) * MC

print(f"=== troll=15% sweep  |  {total} runs  |  v={V} ===")
results = {}; t0 = time.time(); done = 0

for tname, tobj in TRAJS:
    for N in NS:
        rmses = [sim(tobj, N, TROLL, seed=i*31+N+int(TROLL*100), speed_override=V)
                 for i in range(MC)]
        done += MC
        key = f"{tname}_tr{TROLL:.2f}_N{N}"
        results[key] = {
            'rmse_mean': round(float(np.mean(rmses)), 4),
            'rmse_std':  round(float(np.std(rmses)),  4),
            'rmse_ci95': round(float(1.96*np.std(rmses)/np.sqrt(MC)), 4),
            'mc_runs': MC,
        }
        eta = (total-done) / (done/(time.time()-t0)) if done > 0 else 0
        print(f"  {key}: {results[key]['rmse_mean']:.4f} ± {results[key]['rmse_std']:.4f}"
              f"  [{done}/{total}  ETA {eta:.0f}s]")

# ceiling fit
def cm(N, a, b): return a + b / np.sqrt(N)
fits = {}
print()
for tname, _ in TRAJS:
    sub = {k:v for k,v in results.items() if k.startswith(tname)}
    Nv = np.array([int(k.split('N')[1]) for k in sub])
    yv = np.array([v['rmse_mean'] for v in sub.values()])
    try:
        po, _ = curve_fit(cm, Nv, yv, p0=[min(yv), 0.3])
        pr = cm(Nv, *po)
        r2 = 1 - np.sum((yv-pr)**2) / np.sum((yv-np.mean(yv))**2)
        fits[tname] = {'a': round(float(po[0]),4), 'b': round(float(po[1]),4), 'r2': round(float(r2),4)}
        print(f"  ceiling fit [{tname}] troll=15%: R²={r2:.4f}  a={po[0]:.4f}  b={po[1]:.4f}")
    except Exception as e:
        fits[tname] = {'error': str(e)}

elapsed = time.time() - t0
out = {
    'config': {'MC': MC, 'speed': V, 'troll': TROLL, 'Ns': NS,
               'engine': 'run_E2_E3_proper.py (original)'},
    'results': results, 'ceiling_fits': fits,
    'metadata': {'elapsed_sec': round(elapsed,1), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
}
with open('E3_troll15_correct.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: E3_troll15_correct.json  ({elapsed:.1f}s)")
