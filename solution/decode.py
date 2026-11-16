"""Turn per-frame model log-odds into a possession label sequence (3-state Viterbi)."""
import numpy as np

H_, A_, D_ = 0, 1, 2
LET = np.array(["H", "A", "D"])


def emissions(dead_logit, team_logit, temp=1.0):
    ld = -np.logaddexp(0, -dead_logit)           # log p(dead)
    la = -np.logaddexp(0, dead_logit)            # log p(alive)
    lh = -np.logaddexp(0, -team_logit)           # log p(H | alive)
    lA = -np.logaddexp(0, team_logit)
    return temp * np.stack([la + lh, la + lA, ld], axis=1)


def viterbi(emit, lam_team, lam_dead):
    n = len(emit)
    trans = np.array([[0, lam_team, lam_dead], [lam_team, 0, lam_dead], [lam_dead, lam_dead, 0]], float)
    score = emit[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - trans
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


def decode_span(dead_logit, team_logit, temp, lam_team, lam_dead, min_poss=50, min_dead=50):
    em = emissions(dead_logit, team_logit, temp)
    return clean(viterbi(em, lam_team, lam_dead), min_poss, min_dead)


# ----------------------------------------------------------------------------------------------
# Hidden semi-Markov decoder: explicit state durations learned from other matches' labels.
# ----------------------------------------------------------------------------------------------
DMAX = 3000      # longest segment considered (frames); 2 minutes


def fit_duration_model(label_arrays, dmax=DMAX, n_bins=40, smooth=1.5):
    """Per-state log duration density and state-to-state log transition matrix, from label arrays (0/1/2, -1 unknown).

    Durations are histogrammed on log-spaced bins (smoothed), so rare long segments still get sensible mass.
    """
    runs = {0: [], 1: [], 2: []}
    trans = np.ones((3, 3))
    for lab in label_arrays:
        v = lab[lab >= 0]
        if len(v) == 0:
            continue
        ch = np.flatnonzero(np.diff(v)) + 1
        s = np.concatenate(([0], ch))
        e = np.concatenate((ch, [len(v)]))
        for a, b, st in zip(s, e, v[s]):
            runs[int(st)].append(b - a)
        for x, y in zip(v[s][:-1], v[s][1:]):
            trans[x, y] += 1
    edges = np.unique(np.round(np.geomspace(1, dmax + 1, n_bins + 1)).astype(int))
    centers = 0.5 * (edges[:-1] + edges[1:])
    logp = np.full((3, dmax + 1), -30.0)
    d = np.arange(1, dmax + 1)
    for st in range(3):
        r = np.array(runs[st])
        if len(r) == 0:
            continue
        cnt, _ = np.histogram(r, bins=edges)
        cnt = cnt.astype(float)
        k = np.exp(-0.5 * (np.arange(-4, 5) / smooth) ** 2)
        cnt = np.convolve(cnt, k / k.sum(), mode="same") + 1e-3
        dens = cnt / (cnt.sum() * np.diff(edges))                     # per-frame probability within each bin
        idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(dens) - 1)
        logp[st, 1:] = np.log(dens[idx])
    np.fill_diagonal(trans, 0)
    logtrans = np.log(trans / trans.sum(1, keepdims=True).clip(1e-9) + 1e-12)
    np.fill_diagonal(logtrans, -np.inf)
    return dict(logp=logp, logtrans=logtrans)


def hsmm_decode(emit, dur, w_dur=1.0, w_trans=1.0, dmax=DMAX):
    """Best segmentation of `emit` (n, 3 log-potentials) under explicit duration + transition priors."""
    n, S = emit.shape
    logp, logtrans = dur["logp"], dur["logtrans"]
    dmax = min(dmax, logp.shape[1] - 1, n)
    E = np.zeros((n + 1, S))
    E[1:] = np.cumsum(emit, axis=0)
    best = np.full((n + 1, S), -np.inf)
    prevbest = np.full((n + 1, S), -np.inf)
    prevarg = np.zeros((n + 1, S), dtype=np.int8)
    bp = np.zeros((n + 1, S), dtype=np.int32)
    prevbest[0, :] = 0.0
    lt = w_trans * np.where(np.isfinite(logtrans), logtrans, 0.0)
    for t in range(1, n + 1):
        dm = min(dmax, t)
        lo = t - dm
        for s in range(S):
            cand = prevbest[lo:t, s][::-1] + (E[t, s] - E[lo:t, s][::-1]) + w_dur * logp[s, 1:dm + 1]
            k = int(np.argmax(cand))
            best[t, s] = cand[k]
            bp[t, s] = k + 1
        for s in range(S):                                           # best predecessor state for a segment starting at t
            c = [best[t, sp] + lt[sp, s] if sp != s else -np.inf for sp in range(S)]
            j = int(np.argmax(c))
            prevbest[t, s] = c[j]
            prevarg[t, s] = j
    path = np.empty(n, dtype=np.int8)
    t, s = n, int(np.argmax(best[n]))
    while t > 0:
        d = bp[t, s]
        path[t - d:t] = s
        s_prev = prevarg[t - d, s]
        t -= d
        s = s_prev
    return path


def decode_span_hsmm(dead_logit, team_logit, temp, dur, w_dur=1.0, w_trans=1.0, min_poss=0, min_dead=0):
    path = hsmm_decode(emissions(dead_logit, team_logit, temp), dur, w_dur, w_trans)
    return clean(path, min_poss, min_dead) if (min_poss or min_dead) else path
