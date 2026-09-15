#!/usr/bin/env python3
"""
zigzag_lowspeed_probe.py — why the crowd-size effect reverses sign on the zigzag at v = 2 m/s.

Background. In speed_ext_zzfix.json the corrected zigzag at v = 2.0 m/s gives RMSE(N = 5 -> 200)
of -5.0% at tr = 5% but +8.3% and +9.0% at tr = 20% and 40%: the crowd term changes sign. The
pre-correction data (speed_ext.json) showed -0.9/-2.7/-3.6%, so this is a consequence of the
periodic-path correction, not of it.

Design. Two additions to the released grid, both needed to separate the candidate explanations:
  (a) an adversary-free level tr = 0. If the increase comes from adversarial dilution it must
      vanish here; if small crowds are helped by vote noise it should persist as N-dependence.
  (b) a heading-offset level. Each zigzag segment runs at exactly +/-45 deg, which coincides with
      the 8-direction vote grid, so honest votes quantise to a single direction and the consensus
      is locked on-axis. Rotating the whole path by 22.5 deg puts the segments midway between vote
      directions. If on-axis lock is the cause, the reversal should weaken or disappear when
      rotated; if it is unchanged, quantisation is not the mechanism.
Also records the phase-resolved error (reversal windows vs mid-segment) and the realised
adversary count, so the "small crowds get useful dither from adversaries" reading can be checked
against the data rather than asserted.

READS : engine only
WRITES: ../data/zigzag_lowspeed_probe.json
RUN   : cd code ; python zigzag_lowspeed_probe.py          (2 speeds x 2 orientations x 5 tr x 6 N
                                                            x MC 30 = 3,600 runs, ~10 min)
        python zigzag_lowspeed_probe.py --smoke            (MC = 3, pipeline check, ~1 min)
        python zigzag_lowspeed_probe.py --block 0          (one rotation/speed block of four, for
                                                            machines with a short job limit; run
                                                            blocks 0-3, then --merge)
        python zigzag_lowspeed_probe.py --merge            (combine the four block files)
Seed  : SeedSequence([2035, rot_deg, int(v*10), N, int(tr*1000), mc])
"""
import sys, json, time, numpy as np
import adversary_ladder as E
from zigzag_periodic import PeriodicZigzag

SMOKE = '--smoke' in sys.argv
BLOCK = int(sys.argv[sys.argv.index('--block')+1]) if '--block' in sys.argv else None
MERGE = '--merge' in sys.argv
MC = 3 if SMOKE else 30
NS = [5, 10, 20, 50, 100, 200]
TRS = [0.0, 0.05, 0.10, 0.20, 0.40]
SPEEDS = [2.0, 5.0]
ROTS = [0.0, 22.5]                      # path rotation in degrees
SEG = 7.0711                            # zigzag segment length (m)


class RotatedZigzag:
    """Periodic zigzag rotated by `deg` about the origin. Arc length is preserved."""

    def __init__(self, deg):
        self.base = PeriodicZigzag()
        r = np.radians(deg)
        self.R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        self.Rt = self.R.T
        self.L = self.base.L

    def at(self, s):
        return self.R @ self.base.at(s)

    def start(self):
        return self.R @ self.base.start()

    def closest(self, p):
        cp, s = self.base.closest(self.Rt @ np.asarray(p))
        return self.R @ cp, s


def run(tj, N, tr, v, seed):
    """One realisation. Returns RMSE, phase-resolved RMSE, consensus strength, adversary count."""
    rng = np.random.default_rng(seed)
    pos = tj.start(); vel = np.zeros(2); hist = [pos.copy()]
    pang = 0.0; cd = np.array([1.0, 0.0])
    err = np.empty(E.FRAMES); pha = np.empty(E.FRAMES); gam = []
    n_adv = int(round(N*tr))
    for f in range(E.FRAMES):
        if f % E.VOTE_INT == 0:
            di = max(0, len(hist)-1-E.DELAY_F); dp = hist[di]
            _, a0 = tj.closest(dp); lap = tj.at(a0+E.LOOK)
            d = lap-dp; nn = np.linalg.norm(d)
            if nn > 1e-10: d /= nn
            ia = np.degrees(np.arctan2(d[1], d[0]))
            votes = E.gen_votes_adv(ia, pang, tr, N, rng, 'T1', 0.0, None); pang = ia
            bl = E.DIRS[votes].mean(axis=0); g = float(np.linalg.norm(bl)); gam.append(g)
            cd = bl/g if g > 1e-10 else np.array([1.0, 0.0])
        vel += E.SMOOTH*(cd*v-vel); pos = pos+vel*E.DT; hist.append(pos.copy())
        cp, a = tj.closest(pos); err[f] = np.linalg.norm(pos-cp); pha[f] = (a % SEG)/SEG
    e2 = err**2
    edge = (pha < 0.1) | (pha >= 0.9); mid = (pha >= 0.4) & (pha < 0.6)
    return (float(np.sqrt(e2.mean())), float(np.sqrt(e2[edge].mean())), float(np.sqrt(e2[mid].mean())),
            float(np.mean(gam)), n_adv)


if MERGE:
    res = {}
    for b in range(4):
        res.update(json.load(open(f'../data/zigzag_lowspeed_probe_block{b}.json'))['results'])
    t0 = time.time()
else:
    t0 = time.time(); res = {}
    paths = {r: RotatedZigzag(r) for r in ROTS}
    blocks = [(r, v) for r in ROTS for v in SPEEDS]
    todo = [blocks[BLOCK]] if BLOCK is not None else blocks
    for rot, v in todo:
        print(f"\n  rotation {rot:>4.1f} deg, v = {v} m/s")
        print(f"    {'tr':>5}{'N':>6}{'RMSE':>9}{'reversal':>10}{'mid-seg':>9}{'gamma':>8}{'n_adv':>7}")
        for tr in TRS:
            for N in NS:
                out = [run(paths[rot], N, tr, v,
                           int(np.random.SeedSequence([2035, int(rot*10), int(v*10), N, int(tr*1000), i]).generate_state(1)[0]))
                       for i in range(MC)]
                a = np.array([o[:4] for o in out])
                m = a.mean(axis=0); sd = a.std(axis=0)
                res[f"rot{rot}_v{v}_tr{tr:.2f}_N{N}"] = {
                    'rot_deg': rot, 'speed': v, 'tr': tr, 'N': N, 'MC': MC,
                    'rmse_mean': round(float(m[0]), 4), 'rmse_std': round(float(sd[0]), 4),
                    'rmse_sem': round(float(sd[0]/np.sqrt(MC)), 4),
                    'rmse_reversal': round(float(m[1]), 4), 'rmse_midsegment': round(float(m[2]), 4),
                    'gamma_mean': round(float(m[3]), 4), 'n_adversaries': out[0][4]}
                print(f"    {tr:5.0%}{N:6d}{m[0]:9.4f}{m[1]:10.4f}{m[2]:9.4f}{m[3]:8.4f}{out[0][4]:7d}", flush=True)

if BLOCK is not None:
    json.dump({'results': res}, open(f'../data/zigzag_lowspeed_probe_block{BLOCK}.json', 'w'), indent=1)
    print(f"\n  block {BLOCK} -> ../data/zigzag_lowspeed_probe_block{BLOCK}.json"); sys.exit(0)

# ---- summary: crowd-size effect N = 5 -> 200, with a Welch test
from scipy import stats
summ = {}
print(f"\n  crowd-size effect, N = 5 -> 200")
print(f"    {'rot':>5}{'v':>6}{'tr':>6}{'dRMSE%':>9}{'Welch p':>11}{'d reversal%':>12}{'d mid-seg%':>11}")
for rot in ROTS:
    for v in SPEEDS:
        for tr in TRS:
            a = res[f"rot{rot}_v{v}_tr{tr:.2f}_N5"]; b = res[f"rot{rot}_v{v}_tr{tr:.2f}_N200"]
            d = 100*(b['rmse_mean']-a['rmse_mean'])/a['rmse_mean']
            _, p = stats.ttest_ind_from_stats(a['rmse_mean'], a['rmse_std'], MC,
                                              b['rmse_mean'], b['rmse_std'], MC, equal_var=False)
            de = 100*(b['rmse_reversal']-a['rmse_reversal'])/a['rmse_reversal']
            dm = 100*(b['rmse_midsegment']-a['rmse_midsegment'])/a['rmse_midsegment']
            summ[f"rot{rot}_v{v}_tr{tr:.2f}"] = {'delta_pct': round(d, 2), 'welch_p': float(p),
                                                 'delta_reversal_pct': round(de, 2), 'delta_midsegment_pct': round(dm, 2)}
            print(f"    {rot:5.1f}{v:6.1f}{tr:6.0%}{d:+9.2f}{p:11.2e}{de:+12.2f}{dm:+11.2f}")

json.dump({'config': {'Ns': NS, 'trolls': TRS, 'speeds': SPEEDS, 'rotations_deg': ROTS, 'MC': MC,
                      'model': 'T1 c=0 (uniform)', 'segment_m': SEG,
                      'phase_windows': 'reversal 0-10%+90-100%, mid-segment 40-60%',
                      'seed': 'SeedSequence([2035, rot*10, v*10, N, tr*1000, mc])', 'smoke': SMOKE},
           'results': res, 'summary': summ,
           'metadata': {'total_runs': len(res)*MC, 'elapsed_sec': round(time.time()-t0, 1),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}},
          open('../data/zigzag_lowspeed_probe'+('_smoke' if SMOKE else '')+'.json', 'w'), indent=1)
print(f"\n  {len(res)*MC} runs in {time.time()-t0:.0f}s -> ../data/zigzag_lowspeed_probe.json")
