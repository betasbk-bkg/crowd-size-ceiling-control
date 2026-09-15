#!/usr/bin/env python3
"""
rerun_zigzag_periodic.py — Experiment 1: corrected-zigzag re-run of every
zigzag-containing campaign.

WHY
---
The released Zigzag is an open polyline of length 70.71 m. The evaluation
horizon is 65 s at 5 m/s = 325 m = 4.6 path lengths. On reaching the far end
the look-ahead target wraps (at(arc) uses arc % circ) while closest() clamps to
the last segment, so the agent enters a limit cycle on the final teeth.
Measured on the released engine: end reached at t ~= 21 s; the remaining ~44 s
(67-68 % of the horizon) is spent within arc 56-70.7 m and the agent never
returns to the front of the path. Full-horizon RMSE ~= 1.96-2.03 versus ~= 1.42
over the first traversal.

FIX
---
PeriodicZigzag extends the same tooth pattern in +x indefinitely. Local geometry
(segment length 7.0711 m, 90 deg vertex turn, reversal spacing, correction
window 3.26x) is identical to the released class; only the terminal artefact is
removed. Verified bit-for-bit against the released class for the first 1278
frames (21.30 s), diverging first at x = 48.75 m, exactly where the look-ahead
crosses the end of the open path.

DESIGN
------
The engine (adversary_ladder.py) is NOT modified. The trajectory object is
swapped at the call site and every seed is produced by the ORIGINAL
campaign.make_seed with traj_id = 3 ('zigzag'), so each corrected cell sits on
the same random stream as the cell it replaces. Corrected vs released is then a
pure geometry contrast, not a seed contrast.

BLOCKS                                          runs
  1 main            12 N x 10 tr x 3 models   18,000
  2 speed ext       12 N x  3 tr (v = 2.0)     1,800
  3 mechanism        2 N x  2 tr                 200
  4 behaviour panel   2 N x  2 tr x 3 x 5       3,000
  5 lag axis         4 (alpha,tau) x 2 N         400
                                       total  23,400   (~1.2 h measured)

ONE-PASS METRIC
---------------
A first-traversal RMSE is computed alongside the full-horizon RMSE and stored,
but is NOT the reported metric here. A one-pass metric is one alternative
remedy; this study uses the periodic path (fixed 65 s horizon for all four
geometries, evaluation window not endogenous to N). Storing both leaves the
reporting choice open without a second campaign.

The one-pass value comes from a logged replica of the engine loop. Before any
cell is accepted the replica is checked against adversary_ladder.sim on the same
seed and must agree to 1e-12 on full-horizon RMSE; otherwise the run aborts.

TWO INDEPENDENT IMPLEMENTATIONS
-------------------------------
The same correction is realised two ways:

  (A) PeriodicZigzag  - the tooth pattern continued analytically in +x;
                        closest() inspects the three local teeth, O(1).
  (F) Zigzag(ns=NS_F) - the RELEASED class with more segments, so the agent
                        never reaches the end inside the horizon. No new code
                        at all; closest() scans every segment, O(ns).

Measured arc actually reached in 65 s is ~209 m (effective 3.2 m/s after lag
and corner cutting), so ns = 50 (353.6 m) leaves a wide margin. The two give
identical RMSE - max |difference| = 0 over the checked cells - so the corrected
result does not depend on the implementation. (A) is used for the campaign
because it is ~12x faster (0.130 vs 1.332 s/run); (F) is run on a grid subset
as the cross-check and its agreement is recorded in the output.

crosscheck_AF() runs before the campaign and ABORTS on any nonzero difference.

USAGE
  python rerun_zigzag_periodic.py --smoke     # 60 runs, pipeline check only
  python rerun_zigzag_periodic.py             # full 23,400 runs
  python rerun_zigzag_periodic.py --block 1   # a single block

Place next to campaign.py and zigzag_periodic.py inside code/.
Output: data/zigzag_periodic_rerun.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adversary_ladder as E          # noqa: E402
import campaign as C                  # noqa: E402
from zigzag_periodic import PeriodicZigzag   # noqa: E402

TRAJ = 'zigzag'
NS_F = 50            # segment count for implementation (F); 353.6 m >> 209 m reached
OUT = os.path.join(os.path.dirname(HERE), 'data', 'zigzag_periodic_rerun.json')


# --------------------------------------------------------------------------
# logged replica of the engine loop — full-horizon and one-pass in one pass
# --------------------------------------------------------------------------
def sim_logged(traj, N, tr, seed, model='T0', coherence=0.0, speed=5.0):
    """Mirror of adversary_ladder.sim with per-frame arc logging.

    Consumes the RNG in exactly the same order as sim(), so the full-horizon
    RMSE is bit-identical; verified by selftest_replica().
    """
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    pang = 0.0
    cur_dir = np.array([1., 0.])
    errs = np.empty(E.FRAMES)
    arcs = np.empty(E.FRAMES)
    for f in range(E.FRAMES):
        if f % E.VOTE_INT == 0:
            di = max(0, len(hist) - 1 - E.DELAY_F)
            dp = hist[di]
            _, arc = traj.closest(dp)
            lap = traj.at(arc + E.LOOK)
            d = lap - dp
            n = np.linalg.norm(d)
            if n > 1e-10:
                d = d / n
            iang = np.degrees(np.arctan2(d[1], d[0]))
            votes = E.gen_votes_adv(iang, pang, tr, N, rng, model, coherence, None)
            pang = iang
            bl = E.DIRS[votes].mean(axis=0)
            g = np.linalg.norm(bl)
            cur_dir = bl / g if g > 1e-10 else np.array([1., 0.])
        vel = vel + E.SMOOTH * (cur_dir * speed - vel)
        pos = pos + vel * E.DT
        hist.append(pos.copy())
        cp, a = traj.closest(pos)
        errs[f] = np.linalg.norm(pos - cp)
        arcs[f] = a
    full = float(np.sqrt(np.mean(errs ** 2)))
    # one pass = frames until the projected arc first completes one path length
    L = traj.circ
    done = np.where(arcs >= L - 0.5)[0]
    cut = int(done[0]) + 1 if len(done) else E.FRAMES
    onep = float(np.sqrt(np.mean(errs[:cut] ** 2)))
    return {'rmse': full, 'rmse_onepass': onep, 'onepass_frames': cut,
            'completed': bool(len(done))}


def selftest_replica():
    """The replica must reproduce sim() exactly, or the one-pass value is void."""
    z = PeriodicZigzag()
    for (N, tr, mc) in [(5, 0.20, 0), (200, 0.40, 7), (30, 0.05, 3)]:
        seed = C.make_seed(TRAJ, N, tr, 'T0', mc)
        a = E.sim(z, N, tr, seed=seed, model='T1', coherence=0.0, speed=C.SPEED)['rmse']
        b = sim_logged(z, N, tr, seed, model='T1', coherence=0.0, speed=C.SPEED)['rmse']
        if abs(a - b) > 1e-12:
            raise SystemExit(f"replica mismatch at N={N} tr={tr}: {a!r} vs {b!r}")
    print("  replica check: full-horizon RMSE identical to sim() (3 cells, <1e-12)")


def crosscheck_AF(cells=None):
    """(A) periodic vs (F) released class with ns=NS_F. Any difference aborts."""
    z_a = PeriodicZigzag()
    z_f = E.Zigzag(amp=5, ns=NS_F, sx=5)
    cells = cells or [(N, tr, m) for N in (5, 30, 200)
                      for tr in (0.05, 0.20, 0.40) for m in ('T0', 'T1')]
    worst, worst_at = 0.0, None
    for (N, tr, model) in cells:
        seed = C.make_seed(TRAJ, N, tr, model, 0)
        coh = 0.0 if model == 'T0' else 1.0
        ra = E.sim(z_a, N, tr, seed=seed, model='T1', coherence=coh, speed=C.SPEED)['rmse']
        rf = E.sim(z_f, N, tr, seed=seed, model='T1', coherence=coh, speed=C.SPEED)['rmse']
        d = abs(ra - rf)
        if d > worst:
            worst, worst_at = d, (N, tr, model)
    if worst != 0.0:
        raise SystemExit(f"A/F cross-check FAILED: max|diff|={worst:.3e} at {worst_at}")
    print(f"  A/F cross-check: {len(cells)} cells, max|RMSE difference| = 0 "
          f"(periodic vs released class with ns={NS_F})")
    return {'cells': len(cells), 'max_abs_diff': 0.0, 'ns_F': NS_F,
            'note': 'periodic subclass and lengthened released class agree exactly'}


def agg(rm, gm=None, extra=None):
    a = np.array(rm)
    d = {'rmse_mean': round(float(a.mean()), 4),
         'rmse_std': round(float(a.std()), 4),
         'rmse_ci95': round(float(1.96 * a.std() / np.sqrt(len(a))), 4),
         'mc_runs': len(a)}
    if gm is not None:
        d['gamma_mean'] = round(float(np.mean(gm)), 4)
    if extra:
        d.update(extra)
    return d


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
def block1_main(z, mc, ns, trs, res):
    """12 N x 10 tr x 3 models. Seeds and call form identical to campaign.run_one."""
    t0 = time.time()
    n = 0
    for N in ns:
        for tr in trs:
            for model in C.MODELS:
                rm, gm, op = [], [], []
                for i in range(mc):
                    seed = C.make_seed(TRAJ, N, tr, model, i)
                    if model == 'T0':
                        out = E.sim(z, N, tr, seed=seed, model='T1', coherence=0.0, speed=C.SPEED)
                    elif model == 'T1':
                        out = E.sim(z, N, tr, seed=seed, model='T1', coherence=1.0, speed=C.SPEED)
                    else:
                        out = E.sim_ext(z, N, tr, seed=seed, model='T2', coherence=1.0,
                                        speed=C.SPEED, k_hold=C.K_T2)
                    rm.append(out['rmse'])
                    gm.append(out['gamma_mean'])
                    if model == 'T0':
                        op.append(sim_logged(z, N, tr, seed, 'T1', 0.0, C.SPEED)['rmse_onepass'])
                    n += 1
                extra = {'rmse_onepass_mean': round(float(np.mean(op)), 4)} if op else None
                res[f"{TRAJ}_tr{tr:.2f}_N{N}_{model}"] = agg(rm, gm, extra)
    print(f"  [1] main: {n} runs, {time.time()-t0:.0f}s")
    return n


def block2_speed(z, mc, ns, res):
    t0 = time.time()
    n = 0
    for tr in [0.05, 0.20, 0.40]:
        for N in ns:
            rm = []
            for i in range(mc):
                ss = np.random.SeedSequence([C.CAMPAIGN, C.TRAJ_ID[TRAJ], N,
                                             int(round(tr * 1000)), 0, 0, 0,
                                             int(2.0 * 10) * 100000 + i])
                rm.append(E.sim(z, N, tr, seed=int(ss.generate_state(1)[0]),
                                model='T1', coherence=0.0, speed=2.0)['rmse'])
                n += 1
            res[f"{TRAJ}_tr{tr:.2f}_N{N}_v2"] = agg(rm)
    print(f"  [2] speed ext (v=2.0): {n} runs, {time.time()-t0:.0f}s")
    return n


def block3_mech(z, mc, res):
    t0 = time.time()
    n = 0
    for N in [5, 200]:
        for tr in [0.05, 0.20]:
            rm, gm = [], []
            for i in range(mc):
                o = E.sim(z, N, tr, seed=C.make_seed(TRAJ, N, tr, 'T0', i),
                          model='T1', coherence=0.0, speed=5.0)
                rm.append(o['rmse'])
                gm.append(o['gamma_mean'])
                n += 1
            res[f"{TRAJ}_N{N}_tr{tr:.2f}_mech"] = agg(rm, gm)
    print(f"  [3] mechanism: {n} runs, {time.time()-t0:.0f}s")
    return n


def block4_behavior(z, mc, res):
    import campaign_behavior as CB
    t0 = time.time()
    n = 0
    for N in [30, 100]:
        for tr in [0.15, 0.40]:
            for model in ['T0', 'T1', 'T2']:
                for beh in CB.BEHAV:
                    rm = []
                    for i in range(mc):
                        seed = CB.make_seed_b(TRAJ, N, tr, model, beh, i)
                        rm.append(CB.sim_behavior(z, N, tr, seed, model, beh)['rmse'])
                        n += 1
                    res[f"{TRAJ}_N{N}_tr{tr:.2f}_{model}_{beh}"] = agg(rm)
    print(f"  [4] behaviour panel: {n} runs, {time.time()-t0:.0f}s")
    return n


def block5_lagaxis(z, mc, res):
    """Mutates E.SMOOTH / E.DELAY_F; restored in finally."""
    t0 = time.time()
    n = 0
    s0, d0 = E.SMOOTH, E.DELAY_F
    try:
        for (aA, tF) in [(0.20, 13), (0.10, 13), (0.20, 26), (0.10, 26)]:
            E.SMOOTH, E.DELAY_F = aA, tF
            for N in [5, 200]:
                rm = []
                for i in range(mc):
                    ss = np.random.SeedSequence([2026, 9, C.TRAJ_ID[TRAJ], N,
                                                 int(aA * 100), tF, i])
                    rm.append(E.sim(z, N, 0.20, seed=int(ss.generate_state(1)[0]),
                                    model='T1', coherence=0.0, speed=5.0)['rmse'])
                    n += 1
                res[f"{TRAJ}_a{aA}_t{tF}_N{N}"] = agg(rm)
    finally:
        E.SMOOTH, E.DELAY_F = s0, d0
    print(f"  [5] lag axis: {n} runs, {time.time()-t0:.0f}s")
    return n


# --------------------------------------------------------------------------
# G1b gate
# --------------------------------------------------------------------------
def gate_g1b(res, ns):
    """dRMSE(N=5 -> 200) per tr, T0, corrected vs released."""
    old_path = os.path.join(os.path.dirname(HERE), 'data', 'campaign_main_mc50_fixed.json')
    old = json.load(open(old_path))['results'] if os.path.exists(old_path) else {}
    lo, hi = ns[0], ns[-1]
    print("\n=== G1b gate: dRMSE(N=%d -> %d), model T0 ===" % (lo, hi))
    print(f"{'tr':>6}{'corrected':>22}{'released':>20}")
    print(f"{'':>6}{'N=lo':>9}{'N=hi':>7}{'d%':>6}{'N=lo':>9}{'N=hi':>7}{'d%':>6}")
    worst = 0.0
    for tr in sorted({float(k.split('_tr')[1][:4]) for k in res if k.endswith('_T0')}):
        try:
            a = res[f"{TRAJ}_tr{tr:.2f}_N{lo}_T0"]['rmse_mean']
            b = res[f"{TRAJ}_tr{tr:.2f}_N{hi}_T0"]['rmse_mean']
        except KeyError:
            continue
        dn = (a - b) / a * 100
        worst = max(worst, abs(dn))
        oa = old.get(f"{TRAJ}_tr{tr:.2f}_N{lo}_T0", {}).get('rmse_mean')
        ob = old.get(f"{TRAJ}_tr{tr:.2f}_N{hi}_T0", {}).get('rmse_mean')
        od = f"{(oa-ob)/oa*100:6.2f}" if oa and ob else "     -"
        print(f"{tr:6.0%}{a:9.4f}{b:7.4f}{dn:6.2f}"
              f"{oa if oa else 0:9.4f}{ob if ob else 0:7.4f}{od}")
    verdict = ("PASS - keep conclusion, report corrected data" if worst <= 2.0 else
               "REPHRASE - keep conclusion, restate the N-independence wording with numbers"
               if worst <= 5.0 else
               "REWRITE - zigzag conclusion does not survive; extension required")
    print(f"\n  max |dRMSE| = {worst:.2f}%  ->  {verdict}")
    return worst, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--block', type=int, default=0, help='1-5, 0 = all')
    a = ap.parse_args()

    z = PeriodicZigzag()
    print("=== Experiment 1: periodic-zigzag re-run ===")
    print(f"  path length {z.circ:.2f} m (finite window) | periodic in +x | "
          f"engine unmodified | seeds = campaign.make_seed(traj_id={C.TRAJ_ID[TRAJ]})")
    selftest_replica()
    xcheck = crosscheck_AF()

    if a.smoke:
        mc, ns, trs = 3, [5, 200], [0.20, 0.40]
        print("  SMOKE: pipeline check only, results not interpreted")
    else:
        mc, ns, trs = C.MC, C.NS, C.TRS

    res = {}
    t0 = time.time()
    n = 0
    for i, fn in [(1, lambda: block1_main(z, mc, ns, trs, res)),
                  (2, lambda: block2_speed(z, mc, ns, res)),
                  (3, lambda: block3_mech(z, mc, res)),
                  (4, lambda: block4_behavior(z, mc, res)),
                  (5, lambda: block5_lagaxis(z, mc, res))]:
        if a.block in (0, i):
            n += fn()

    worst, verdict = gate_g1b(res, ns)

    payload = {
        'config': {'MC': mc, 'Ns': ns, 'trolls': trs, 'models': C.MODELS,
                   'trajectory': 'PeriodicZigzag (corrected)',
                   'engine': 'adversary_ladder.py (unmodified)',
                   'seed': 'campaign.make_seed / speed+lagaxis conventions as released',
                   'metric': 'full-horizon RMSE; rmse_onepass stored for T0 but not reported',
                   'smoke': a.smoke},
        'results': res,
        'crosscheck_A_vs_F': xcheck,
        'gate_g1b': {'max_abs_delta_pct': round(worst, 3), 'verdict': verdict},
        'metadata': {'total_runs': n, 'elapsed_sec': round(time.time() - t0, 1),
                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fn = OUT.replace('.json', '_smoke.json') if a.smoke else OUT
    json.dump(payload, open(fn, 'w'), indent=2)
    print(f"\n  {n} runs in {time.time()-t0:.0f}s -> {fn}")


if __name__ == '__main__':
    main()
