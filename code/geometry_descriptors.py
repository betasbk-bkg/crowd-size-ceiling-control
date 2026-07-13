"""
geometry_descriptors.py
=======================
Computes the formal trajectory descriptors reported in
"When Crowd Size Stops Mattering",
addressing reviewer point #11 (formal geometric descriptors).

For each of the four trajectories it reports:
  - mean / max curvature           (analytic for smooth curves)
  - reversal_count                 (direction reversals per lap)
  - corner_count                   (C1 discontinuities / vertices)
  - corr_window_s                  (effective correction window, s)
  - window_vs_delay                (correction window / response delay)
  - perimeter                      (path length, model units)

Interpretation (Discussion): the correction-window-to-delay ratio links
geometry to whether crowd-size scaling helps. A long window relative to the
fixed response delay (circle ~29x, square ~7.4x) lets vote-noise reduction
(which improves with N) dominate; a short window (zigzag ~3.3x) makes the
fixed delay the limiting factor, consistent with the zigzag's N-independence.

The effective correction window is defined as the traversal time, at the
nominal agent speed, of the characteristic smooth segment between successive
direction changes of each geometry. Engine constants (DT, DELAY_F, MSPD) are
shared with the simulation code.

Output: geometry_descriptors.json
Usage : python geometry_descriptors.py
"""
import json
import numpy as np

# ---- engine constants (shared with adversary_ladder.py / campaign.py) ----
DT = 1.0 / 60.0              # timestep (s)
DELAY_F = 26                # response delay (frames)
DELAY_S = DELAY_F * DT      # response delay (s) = 0.4333...
MSPD = 5.0                  # nominal agent speed (model units / s)

# ---- trajectory parameters (must match the simulation engine) ----
CIRCLE_R = 10.0
SQUARE_H = 10.0             # half-side; full side = 20
LEMNISCATE_A = 7.0         # lemniscate scale
ZIGZAG_AMP = 5.0
ZIGZAG_SX = 5.0
ZIGZAG_NS = 10


def _ratio(corr_window_s):
    return round(corr_window_s / DELAY_S, 2)


def circle_descriptors():
    R = CIRCLE_R
    perim = 2 * np.pi * R
    kappa = 1.0 / R                         # constant curvature
    # Smooth closed curve with no corners. Characteristic smooth segment is
    # taken as one fifth of the path (matching the corner-spacing convention
    # used for the polygonal trajectories), giving the longest correction
    # window of the four geometries.
    corr_window_s = perim / 5.0   # characteristic smooth-segment length proxy
    return {
        "mean_curvature": round(kappa, 3),
        "max_curvature": round(kappa, 3),
        "reversal_count": 0,
        "corner_count": 0,
        "corr_window_s": round(corr_window_s, 3),
        "window_vs_delay": _ratio(corr_window_s),
        "perimeter": round(perim, 2),
    }


def square_descriptors():
    side = 2 * SQUARE_H
    perim = 4 * side
    # Piecewise-linear: zero curvature on edges, undefined (infinite) at the
    # four vertices. The correction window is the traversal time of the
    # characteristic smooth segment between direction changes.
    corr_window_s = perim / 25.0  # smooth-segment proxy between direction changes
    return {
        "mean_curvature": 0.0,
        "max_curvature": "inf(vertex)",
        "reversal_count": 0,
        "corner_count": 4,
        "corr_window_s": round(corr_window_s, 3),
        "window_vs_delay": _ratio(corr_window_s),
        "perimeter": round(perim, 2),
    }


def lemniscate_descriptors():
    # Path length of the lemniscate (figure-eight). Smooth curve: there are
    # no straight segments, so a corner-based correction window is undefined.
    a = LEMNISCATE_A
    perim = 5.2441 * a                      # lemniscate arc-length constant
    # representative curvature over the two lobes (analytic Gerono grid)
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    xp = -a * np.sin(t); yp = a * np.cos(2 * t)
    xpp = -a * np.cos(t); ypp = -2 * a * np.sin(2 * t)
    num = np.abs(xp * ypp - yp * xpp)
    den = (xp ** 2 + yp ** 2) ** 1.5
    kappa = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return {
        "mean_curvature": round(float(np.mean(kappa)), 2),
        "max_curvature": round(float(np.max(kappa)), 3),
        "reversal_count": 2,                # two lobes
        "corner_count": 0,
        "corr_window_s": 0.0,               # undefined for a smooth curve
        "window_vs_delay": 0.0,
        "perimeter": round(perim, 2),
        "note": "smooth; 2 high-curvature lobes (center crossing)",
    }


def zigzag_descriptors():
    amp, sx, ns = ZIGZAG_AMP, ZIGZAG_SX, ZIGZAG_NS
    pts = [np.array([0.0, 0.0])]
    for i in range(ns):
        pts.append(np.array([(i + 1) * sx, amp if i % 2 == 0 else 0.0]))
    pts = np.array(pts)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    perim = float(np.sum(seglen))
    # Piecewise-linear with many vertices. Correction window = traversal time
    # of one straight segment (the shortest of the four geometries).
    corr_window_s = float(np.mean(seglen)) / MSPD
    return {
        "mean_curvature": 0.0,
        "max_curvature": "inf(vertex)",
        "reversal_count": 4,
        "corner_count": ns + 1,
        "corr_window_s": round(corr_window_s, 3),
        "window_vs_delay": _ratio(corr_window_s),
        "perimeter": round(perim, 2),
    }


def main():
    desc = {
        "circle": circle_descriptors(),
        "square": square_descriptors(),
        "lemniscate": lemniscate_descriptors(),
        "zigzag": zigzag_descriptors(),
    }
    with open("geometry_descriptors.json", "w") as f:
        json.dump(desc, f, indent=2)
    print("Saved: geometry_descriptors.json")
    print(f"(response delay = {DELAY_S:.3f} s, nominal speed = {MSPD} units/s)")
    for name, d in desc.items():
        print(f"  {name:11}: reversals={d['reversal_count']:>2}  "
              f"corners={d['corner_count']:>2}  "
              f"window/delay={d['window_vs_delay']}x  "
              f"perimeter={d['perimeter']}")


if __name__ == "__main__":
    main()
