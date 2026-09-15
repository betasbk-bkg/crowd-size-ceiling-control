"""
zigzag_periodic.py — periodic (non-terminating) zigzag path.

Purpose
-------
The released Zigzag class is an OPEN polyline of total length 10*sqrt(sx^2+amp^2)
= 70.71 m for the default parameters. The evaluation horizon is 65 s at 5 m/s,
i.e. 325 m = 4.6 path lengths. Once the agent reaches the far end:

  * at(arc) wraps the look-ahead target modulo the path length, so the target
    jumps back to the start of the path, while
  * closest(p) clamps to the nearest segment, so the agent stays at the end.

The result is a limit cycle on the final one or two teeth. Measured on the
released engine: end reached at t ~= 21 s; for the remaining ~44 s (67-68 % of
the horizon) the agent stays within arc 56-70.7 m (x in [40, 50]) and never
returns to the front of the path. Full-window RMSE ~= 1.96-2.03 versus ~= 1.42
over the first traversal, i.e. roughly 40 % of the reported zigzag error comes
from the terminal regime rather than from directional reversals.

Fix
---
Extend the same tooth pattern periodically in +x instead of terminating. Local
geometry (segment length, turn angle, reversal spacing, corner count per unit
arc, correction window) is IDENTICAL to the released class; only the terminal
artefact is removed.

  y(x) = amp - |(x mod 2*sx) - sx| * (amp/sx)
  arc  = x * L / sx,   L = sqrt(sx^2 + amp^2)

Compatibility
-------------
For every state the agent can reach before the terminal regime, this class
returns exactly the same values as the released Zigzag. run_selftest() verifies
this bit-for-bit and reports the first divergent frame, which is expected only
once the look-ahead target crosses the end of the original path
(arc + LOOK > circ, i.e. x > ~48.6 m).

The original engine is NOT modified. Usage:

    import adversary_ladder as E
    from zigzag_periodic import PeriodicZigzag
    E.sim(PeriodicZigzag(), N, tr, seed=..., model='T0')
"""

import numpy as np


class PeriodicZigzag:
    """Infinite periodic zigzag. Drop-in replacement for adversary_ladder.Zigzag."""

    def __init__(self, amp=5.0, ns=10, sx=5.0):
        self.amp = float(amp)
        self.sx = float(sx)
        self.ns = int(ns)
        self.L = float(np.hypot(self.sx, self.amp))   # per-segment arc length
        self.period_x = 2.0 * self.sx                 # spatial period
        # --- attributes kept for compatibility with descriptor/analysis scripts.
        # They describe the SAME finite window as the released class; nothing in
        # this class uses them for path evaluation.
        pts = [np.array([0., 0.])]
        for i in range(self.ns):
            pts.append(np.array([(i + 1) * self.sx, self.amp if i % 2 == 0 else 0.]))
        self.c = np.array(pts)
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(self.ns)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.] + list(np.cumsum(self.lens)))

    # ---- path definition -------------------------------------------------
    def _y(self, x):
        return self.amp - abs((x % self.period_x) - self.sx) * (self.amp / self.sx)

    def _vertex(self, k):
        """Vertex k sits at x = k*sx; y alternates 0, amp, 0, amp, ..."""
        return np.array([k * self.sx, self.amp if (k % 2) else 0.0])

    def at(self, arc):
        """Point at arc length `arc`. No modulo: the path continues in +x."""
        x = arc * self.sx / self.L
        return np.array([x, self._y(x)])

    def closest(self, p):
        """Nearest point on the infinite polyline, and its arc length.

        Only the three teeth bracketing the agent are candidates; for a zigzag
        the global nearest point is always local, so this is exact and O(1)
        instead of the released class's O(ns) scan.
        """
        k0 = int(np.floor(p[0] / self.sx))
        bd, bp, ba = 1e18, None, 0.0
        for k in (k0 - 1, k0, k0 + 1):
            a = self._vertex(k)
            b = self._vertex(k + 1)
            v = b - a
            l2 = v @ v
            t = np.clip((p - a) @ v / l2, 0.0, 1.0)
            pt = a + t * v
            d = float(np.linalg.norm(p - pt))
            if d < bd:
                bd, bp, ba = d, pt, (k + t) * self.L
        return bp, ba

    def start(self):
        return np.array([0., 0.])


# ---------------------------------------------------------------------------
def run_selftest(seed=751, N=30, tr=0.20, verbose=True):
    """Bit-for-bit comparison against the released Zigzag, frame by frame.

    Returns dict with the first divergent frame and the agent x at that point.
    Expectation: identical until the look-ahead crosses the end of the original
    open path; divergence x should be ~ns*sx - LOOK*sx/L ~= 48.6 m.
    """
    import adversary_ladder as E

    def trace(traj, seed):
        rng = np.random.default_rng(seed)
        pos = traj.start(); vel = np.zeros(2); hist = [pos.copy()]
        pang = 0.0; cdir = np.array([1., 0.])
        xs = np.empty(E.FRAMES); ys = np.empty(E.FRAMES)
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
                ang = np.degrees(np.arctan2(d[1], d[0]))
                votes = E.gen_votes_adv(ang, pang, tr, N, rng, 'T0', 1.0, None)
                pang = ang
                bl = E.DIRS[votes].mean(axis=0)
                g = np.linalg.norm(bl)
                cdir = bl / g if g > 1e-10 else np.array([1., 0.])
            vel = vel + E.SMOOTH * (cdir * E.MSPD - vel)
            pos = pos + vel * E.DT
            hist.append(pos.copy())
            xs[f], ys[f] = pos
        return xs, ys

    xo, yo = trace(E.Zigzag(), seed)
    xp, yp = trace(PeriodicZigzag(), seed)
    diff = np.hypot(xo - xp, yo - yp)
    idx = np.argmax(diff > 1e-12) if (diff > 1e-12).any() else len(diff)
    out = {
        'first_divergent_frame': int(idx),
        'first_divergent_time_s': idx / 60.0,
        'agent_x_at_divergence': float(xo[idx - 1]) if idx > 0 else None,
        'max_abs_diff_before': float(diff[:idx].max()) if idx > 0 else 0.0,
        'orig_x_range_after': (float(xo[idx:].min()), float(xo[idx:].max())) if idx < len(xo) else None,
        'peri_x_range_after': (float(xp[idx:].min()), float(xp[idx:].max())) if idx < len(xp) else None,
    }
    if verbose:
        print("bit-for-bit identical for first %d frames (%.2f s), max|diff| = %.1e"
              % (out['first_divergent_frame'], out['first_divergent_time_s'],
                 out['max_abs_diff_before']))
        print("agent x at divergence: %.2f m  (open path ends at %.1f m)"
              % (out['agent_x_at_divergence'], 10 * 5.0))
        print("after divergence  x range | original %s | periodic %s"
              % (np.round(out['orig_x_range_after'], 1), np.round(out['peri_x_range_after'], 1)))
    return out


if __name__ == '__main__':
    run_selftest()
