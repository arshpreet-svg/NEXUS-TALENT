"""Per-frame features computed identically for the provided game (ATD) and for other games.

Input: players P (n, slots, 2) in metres, ball B (n, 2) in metres, team (slots,) 0=home 1=away.
Only geometry is used, so the same code runs on any tracking source.
"""
import numpy as np

FPS = 25
L, W = 105.3, 68.0
CAP = 60.0


def _cum(x):
    return np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))


def roll_mean(x, w):
    """Centred NaN-ignoring rolling mean, window w frames (odd)."""
    n = len(x)
    ok = np.isfinite(x)
    cs, cn = _cum(np.where(ok, x, 0.0)), _cum(ok.astype(float))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - w // 2), np.minimum(n, i + w // 2 + 1)
    c = cn[hi] - cn[lo]
    with np.errstate(all="ignore"):
        return np.where(c > 0, (cs[hi] - cs[lo]) / c, np.nan)


def roll_median(x, w, chunk=20000):
    import warnings
    from numpy.lib.stride_tricks import sliding_window_view
    n, pad = len(x), w // 2
    xp = np.pad(x.astype(np.float64), (pad, w - 1 - pad), constant_values=np.nan)
    out = np.empty(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in range(0, n, chunk):
            win = sliding_window_view(xp[a:a + chunk + w - 1], w)
            out[a:a + len(win)] = np.nanmedian(win, axis=1)
    return out


def roll_ext(x, w, fn):
    """Rolling min/max ignoring NaN, chunked."""
    import warnings
    from numpy.lib.stride_tricks import sliding_window_view
    n, pad = len(x), w // 2
    xp = np.pad(x.astype(np.float64), (pad, w - 1 - pad), constant_values=np.nan)
    out = np.empty(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in range(0, n, 20000):
            win = sliding_window_view(xp[a:a + 20000 + w - 1], w)
            out[a:a + len(win)] = fn(win, axis=1)
    return out


def fill_ball(B, max_gap):
    n = len(B)
    obs = np.isfinite(B[:, 0])
    out = B.astype(np.float64).copy()
    idx = np.flatnonzero(obs)
    gaps = np.diff(idx)
    for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
        if g - 1 <= max_gap:
            t = np.arange(1, g)
            for c in (0, 1):
                out[a + t, c] = out[a, c] + (out[a + g, c] - out[a, c]) * t / g
    return out, obs


def slot_speed(P, lag=12):
    """Speed per slot (m/s) from positions, over a +-lag frame baseline."""
    n = len(P)
    S = np.full((n, P.shape[1]), np.nan)
    d = np.hypot(P[2 * lag:, :, 0] - P[:-2 * lag, :, 0], P[2 * lag:, :, 1] - P[:-2 * lag, :, 1])
    S[lag:n - lag] = d / (2 * lag / FPS)
    return S


def attack_dir(P, team, segments=None):
    """+1 where the HOME team attacks toward +x, -1 toward -x, per frame. A team's side is inferred from where its
    players stand on average over each segment (a half): the team whose mean x is smaller defends the left goal."""
    n = len(P)
    segments = segments or [(0, n - 1)]
    out = np.ones(n)
    import warnings
    for a, b in segments:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mh = np.nanmean(P[a:b + 1][:, team == 0, 0])
            ma = np.nanmean(P[a:b + 1][:, team == 1, 0])
        d = 1.0 if ma > mh else -1.0       # away team stands further right => home attacks right
        lo = 0 if a == segments[0][0] else a
        out[lo:b + 1] = d
    # frames after the last segment keep the last direction
    out[segments[-1][1] + 1:] = out[segments[-1][1]]
    return out


def features(P, B, team, fill_s=1.0, segments=None):
    P = P[:, :, :2].astype(np.float64)
    n = len(B)
    ball, obs = fill_ball(B, int(fill_s * FPS))
    have = np.isfinite(ball[:, 0])
    D = np.hypot(P[:, :, 0] - ball[:, [0]], P[:, :, 1] - ball[:, [1]])
    D = np.where(np.isfinite(D), D, 1e9)
    isH = (team == 0)[None, :]
    DH, DA = np.where(isH, D, 1e9), np.where(~isH, D, 1e9)
    sH, sA = np.sort(DH, axis=1)[:, :2], np.sort(DA, axis=1)[:, :2]
    f = {}
    capd = lambda a: np.where((a < 1e8) & have[:, None] if a.ndim > 1 else (a < 1e8) & have, np.minimum(a, CAP), np.nan)
    f["dH1"], f["dH2"] = capd(sH[:, 0]), capd(sH[:, 1])
    f["dA1"], f["dA2"] = capd(sA[:, 0]), capd(sA[:, 1])
    f["adv"] = f["dA1"] - f["dH1"]                          # >0 : home closer
    for r in (3, 8, 15):
        f["nH%d" % r] = np.where(have, (DH < r).sum(1), np.nan)
        f["nA%d" % r] = np.where(have, (DA < r).sum(1), np.nan)
    dmin = np.fmin(f["dH1"], f["dA1"])
    f["dmin"] = dmin
    # ball kinematics from the filled track
    lag = 6
    bs = np.full(n, np.nan)
    bs[lag:n - lag] = np.hypot(ball[2 * lag:, 0] - ball[:-2 * lag, 0], ball[2 * lag:, 1] - ball[:-2 * lag, 1]) / (2 * lag / FPS)
    f["bspeed"] = bs
    bd = np.where(have, np.minimum.reduce([ball[:, 0], L - ball[:, 0], ball[:, 1], W - ball[:, 1]]), np.nan)
    f["bedge"] = bd
    f["obs"] = obs.astype(float)
    # collective motion
    S = slot_speed(P)
    f["pspd"] = np.nanmedian(S, axis=1) if True else None
    f["pstill"] = np.nanmean(S < 0.6, axis=1)
    f["nplay"] = np.isfinite(P[:, :, 0]).sum(1).astype(float)
    # team shape (orientation-free): the side in possession tends to be stretched, the other compact
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shp = {}
        for nm, mk in (("H", team == 0), ("A", team == 1)):
            x, y = P[:, mk, 0], P[:, mk, 1]
            cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
            shp["sx" + nm], shp["sy" + nm] = np.nanstd(x, 1), np.nanstd(y, 1)
            shp["cb" + nm] = np.where(have, np.hypot(cx - ball[:, 0], cy - ball[:, 1]), np.nan)
            shp["tspd" + nm] = np.nanmean(S[:, mk], 1)
        f["d_sx"], f["d_sy"] = shp["sxH"] - shp["sxA"], shp["syH"] - shp["syA"]
        f["d_cb"], f["d_tspd"] = shp["cbH"] - shp["cbA"], shp["tspdH"] - shp["tspdA"]
        # pressure on the player nearest the ball, and that player's own speed
        nearest = np.argmin(D, axis=1)
        near_d = D[np.arange(n), nearest]
        near_team = team[nearest]
        pl = P[:, :, :2]
        opp_d = np.full(n, np.nan)
        for t in (0, 1):
            sel = have & (near_d < 1e8) & (near_team == t)
            if sel.any():
                dd = np.hypot(pl[sel][:, :, 0] - pl[sel, nearest[sel], 0][:, None],
                              pl[sel][:, :, 1] - pl[sel, nearest[sel], 1][:, None])
                dd = np.where((team != t)[None, :], dd, np.nan)
                opp_d[sel] = np.nanmin(dd, axis=1)
        f["press"] = np.minimum(opp_d, CAP)
        f["near_spd"] = np.where(have & (near_d < 1e8), S[np.arange(n), nearest], np.nan)
        f["near_is_home"] = np.where(have & (near_d < 1e8), (near_team == 0).astype(float), np.nan)
    for k in ("d_sx", "d_sy", "d_cb", "d_tspd"):
        for name, w in (("3", 75), ("6", 151), ("15", 375)):
            f[k + "_m" + name] = roll_mean(f[k], w | 1)
    f["press_m3"] = roll_mean(f["press"], 75)
    f["near_home_m3"] = roll_mean(f["near_is_home"], 75)
    f["near_home_m6"] = roll_mean(f["near_is_home"], 151)

    # field-position features, in each team's own attacking direction (symmetric under a team swap)
    dirH = attack_dir(P, team, segments)
    progH = lambda x: np.where((dirH > 0).reshape((-1,) + (1,) * (x.ndim - 1)), x, L - x) / L   # 0 = own goal line, 1 = opponent's, for HOME
    f["bprog"] = np.where(have, progH(ball[:, 0]) - 0.5, np.nan)  # >0 : ball is in the away half (away's defending end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, mk, sgn in (("H", team == 0, 1), ("A", team == 1, -1)):
            px = P[:, mk, 0]
            pr = progH(px) if sgn > 0 else 1 - progH(px)                 # team progress toward the goal it attacks
            f["cprog" + nm] = np.nanmean(pr, 1)
            f["line_lo" + nm] = np.nanpercentile(pr, 25, axis=1)
            f["line_hi" + nm] = np.nanpercentile(pr, 75, axis=1)
    f["d_cprog"] = f["cprogH"] - f["cprogA"]
    f["d_linelo"] = f["line_loH"] - f["line_loA"]
    f["d_linehi"] = f["line_hiH"] - f["line_hiA"]
    for k in ("d_cprog", "d_linelo", "d_linehi", "bprog"):
        for name, w in (("3", 75), ("6", 151), ("15", 375)):
            f[k + "_m" + name] = roll_mean(f[k], w | 1)
    for k in ("cprogH", "cprogA", "line_loH", "line_loA", "line_hiH", "line_hiA"):
        del f[k]

    # temporal context at several scales (all offline / centred)
    for name, w in (("1", 25), ("3", 75), ("6", 151), ("15", 375)):
        w = w | 1
        f["bspeed_m" + name] = roll_mean(bs, w)
        f["obs_m" + name] = roll_mean(f["obs"], w)
        f["pspd_m" + name] = roll_median(f["pspd"], w) if w <= 151 else roll_mean(f["pspd"], w)
        f["pstill_m" + name] = roll_mean(f["pstill"], w)
        f["dmin_m" + name] = roll_mean(dmin, w)
        f["adv_m" + name] = roll_mean(np.clip(f["adv"], -15, 15), w)
        f["homeclose_m" + name] = roll_mean(np.where(np.isfinite(f["adv"]), (f["adv"] > 0).astype(float), np.nan), w)
        f["nH8_m" + name] = roll_mean(f["nH8"], w)
        f["nA8_m" + name] = roll_mean(f["nA8"], w)
    f["bedge_min3"] = roll_ext(bd, 75, np.nanmin)
    f["bedge_min6"] = roll_ext(bd, 151, np.nanmin)
    f["out_m3"] = roll_mean((bd < 0).astype(float), 75)
    # ball-free window of the last/next seconds: how long since/until the ball was last seen
    seen_idx = np.where(obs, np.arange(n), -10**9)
    last = np.maximum.accumulate(seen_idx)
    nxt = np.where(obs, np.arange(n), 10**9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    f["since_ball"] = np.minimum((np.arange(n) - last) / FPS, 60)
    f["until_ball"] = np.minimum((nxt - np.arange(n)) / FPS, 60)
    names = sorted(f)
    X = np.stack([f[k] for k in names], axis=1).astype(np.float32)
    return X, names
