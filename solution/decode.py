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
