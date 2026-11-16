"""Possession labelling: per-frame evidence -> 3-state Viterbi (H, A, D) -> segment clean-up.

No labels of the provided game are used anywhere; every constant is a physical/football prior.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

FPS = 25
L, W = 105.3, 68.0
HALVES = [(11185, 90651), (110808, 190197)]   # published in DATA.md
H_, A_, D_ = 0, 1, 2
LET = np.array(["H", "A", "D"])

DEFAULT = dict(
    fill_s=1.0,        # linear-interpolate ball gaps up to this long
    ctrl_scale=1.5,    # metres: distance advantage that makes control "certain"
    ctrl_reach=4.0,    # metres: nearest player farther than this => ball in flight, weak evidence
    speed_win_s=5.0,   # rolling window for collective player speed
    speed_mid=1.6,     # m/s: collective speed separating stoppage from play
    speed_scale=0.4,
    w_ctrl=1.0,
    w_dead=1.0,
    lam_team=40.0,     # Viterbi cost to switch H<->A  (in evidence units)
    lam_dead=40.0,     # cost to enter/leave D
    min_poss_s=2.0,    # shorter possessions are absorbed by neighbours
    min_dead_s=2.0,
)


def roll(x, w, fn=np.nanmedian):
    """Centred rolling stat that ignores NaN, via stride tricks on a padded copy."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = w // 2
    xp = np.pad(x.astype(np.float64), (pad, w - 1 - pad), constant_values=np.nan)
    win = sliding_window_view(xp, w)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(win, axis=1)


def fill_ball(B, max_gap):
    """Interpolate short ball gaps. Returns (ball, observed_mask)."""
    n = len(B)
    obs = np.isfinite(B[:, 0])
    out = B.copy().astype(np.float64)
    idx = np.flatnonzero(obs)
    gaps = np.diff(idx)
    for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
        if g - 1 <= max_gap:
            t = np.arange(1, g)
            for c in (0, 1):
                out[a + t, c] = out[a, c] + (out[a + g, c] - out[a, c]) * t / g
    return out, obs


def evidence(P, B, team, prm):
    n = len(B)
    ball, obs = fill_ball(B, int(prm["fill_s"] * FPS))
    have = np.isfinite(ball[:, 0])
    px, py = P[:, :, 0].astype(np.float64), P[:, :, 1].astype(np.float64)
    d = np.hypot(px - ball[:, [0]], py - ball[:, [1]])          # (n, slots), NaN where missing
    dH = np.nanmin(np.where(team[None, :] == 0, d, np.nan), axis=1) if False else None
    big = np.where(np.isfinite(d), d, 1e9)
    dH = np.where(team[None, :] == 0, big, 1e9).min(1)
    dA = np.where(team[None, :] == 1, big, 1e9).min(1)
    dmin = np.minimum(dH, dA)
    valid = have & (dmin < 1e8)
    ctrl = np.zeros(n)
    w = np.exp(-(np.minimum(dmin, 50) / prm["ctrl_reach"]) ** 2)
    adv = np.tanh((np.minimum(dA, 50) - np.minimum(dH, 50)) / prm["ctrl_scale"])
    ctrl[valid] = (w * adv)[valid] * np.where(obs, 1.0, 0.6)[valid]

    # dead-ball evidence (positive => dead)
    spd = np.nanmean(P[:, :, 2], axis=1)
    m = roll(spd, int(prm["speed_win_s"] * FPS) | 1)
    m = np.where(np.isfinite(m), m, prm["speed_mid"])
    dead = (prm["speed_mid"] - m) / prm["speed_scale"]
    out = have & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
    dead = dead + 2.0 * roll(out.astype(float), 25, np.nanmean)
    return ctrl, dead, dict(ball=ball, obs=obs, dH=dH, dA=dA, spd=m)


def viterbi(emit, lam_team, lam_dead):
    """emit: (n,3) log-potentials. Switch costs are symmetric."""
    n = len(emit)
    trans = np.array([[0, lam_team, lam_dead],
                      [lam_team, 0, lam_dead],
                      [lam_dead, lam_dead, 0]], dtype=np.float64)
    score = emit[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - trans            # cand[i,j]: from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(path):
    ch = np.flatnonzero(np.diff(path)) + 1
    s = np.concatenate(([0], ch))
    e = np.concatenate((ch, [len(path)]))
    return list(zip(s, e, path[s]))


def clean(path, min_poss, min_dead):
    """Absorb very short runs into the longer neighbour (repeat until stable)."""
    path = path.copy()
    for _ in range(10):
        r = runs(path)
        changed = False
        for i, (s, e, v) in enumerate(r):
            lim = min_dead if v == D_ else min_poss
            if e - s < lim and 0 < i < len(r) - 1:
                a, b = r[i - 1], r[i + 1]
                path[s:e] = a[2] if (a[1] - a[0]) >= (b[1] - b[0]) else b[2]
                changed = True
        if not changed:
            break
    return path


def decode(P, B, team, prm=None):
    prm = {**DEFAULT, **(prm or {})}
    ctrl, dead, aux = evidence(P, B, team, prm)
    n = len(B)
    labels = np.full(n, "X", dtype="<U1")
    for lo, hi in HALVES:
        sl = slice(lo, hi + 1)
        em = np.zeros((hi + 1 - lo, 3))
        em[:, H_] = prm["w_ctrl"] * ctrl[sl]
        em[:, A_] = -prm["w_ctrl"] * ctrl[sl]
        em[:, D_] = prm["w_dead"] * dead[sl]
        path = viterbi(em, prm["lam_team"], prm["lam_dead"])
        path = clean(path, int(prm["min_poss_s"] * FPS), int(prm["min_dead_s"] * FPS))
        labels[sl] = LET[path]
    return labels, aux


def write_csv(path, labels):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args()
    prm = {k: float(v) for k, v in (s.split("=") for s in a.set)}
    P, B, team = data.load()
    lab, _ = decode(P, B, team, prm)
    write_csv(a.out, lab)
    print("wrote", a.out)
