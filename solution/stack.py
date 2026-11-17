"""Second-stage (stacked) model on the first-stage logits.

The first stage scores each frame from that frame's geometry. Possession, however, is a temporal object: who had
the ball over the last and next few seconds is strong evidence for who has it now, and how the team evidence is
trending tells where a possession starts or ends. Stage two sees, for every frame, the first-stage dead-ball and
team logits summarised over +/-1, 3, 6 and 15 s (centred, past and future means, slope, extrema) and re-predicts
both quantities.

Leakage rule: stage two is trained ONLY on out-of-fold first-stage logits (each training match scored by a
first-stage model that never saw it), never on in-sample predictions, which are over-confident.

Team symmetry: every team feature is odd under swapping the teams (tl -> -tl), so the team model is trained on the
original and the mirrored copy and its output is symmetrised, exactly like the first stage.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

# Team-neutral first-stage features passed through as context: how much to trust the team evidence depends on
# whether the ball is tracked and how play is moving. All are unchanged by swapping the teams.
CONTEXT = ("obs", "obs_m3", "obs_m15", "since_ball", "until_ball", "pspd_m3", "pstill_m3", "bspeed_m1", "dmin")
WINDOWS = (25, 75, 150, 375)          # half-widths in frames at 25 fps: 1, 3, 6, 15 s
SUB = 5
HP = dict(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def _cs(x):
    return np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))


def _win_mean(c, n, lo, hi):
    """Mean of x[t+lo : t+hi+1] for every t, windows clipped to the sequence."""
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    return ((c[b] - c[a]) / np.maximum(b - a, 1)).astype(np.float32)


def context(X, names):
    """The CONTEXT columns of a first-stage feature matrix."""
    names = list(names)
    return X[:, [names.index(c) for c in CONTEXT]].astype(np.float32)


def features(dl, tl, C=None):
    """Stage-two features for one sequence (len n); C = context(X, names) or None. Returns (n, k) float32, names."""
    n = len(dl)
    cd, ct = _cs(dl), _cs(tl)
    cols, names = [dl.astype(np.float32), tl.astype(np.float32)], ["dl", "tl"]
    for w in WINDOWS:
        tc, tp, tf = _win_mean(ct, n, -w, w), _win_mean(ct, n, -w, 0), _win_mean(ct, n, 0, w)
        dc, dp, df = _win_mean(cd, n, -w, w), _win_mean(cd, n, -w, 0), _win_mean(cd, n, 0, w)
        cols += [tc, tp, tf, tf - tp, dc, dp, df, df - dp,
                 maximum_filter1d(tl, 2 * w + 1).astype(np.float32),
                 minimum_filter1d(tl, 2 * w + 1).astype(np.float32),
                 maximum_filter1d(dl, 2 * w + 1).astype(np.float32)]
        names += ["%s_%d" % (k, w) for k in ("tc", "tp", "tf", "tslope", "dc", "dp", "df", "dslope", "tmax", "tmin", "dmax")]
    F = np.stack(cols, 1)
    if C is not None:
        F, names = np.concatenate([F, C], 1), names + list(CONTEXT)
    return F, names


def _clf(hp):
    return HistGradientBoostingClassifier(**hp)


def _logit(m, X):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Stage2:
    def __init__(self, hp=None, sub=SUB):
        self.hp, self.sub = {**HP, **(hp or {})}, sub

    def fit(self, oof):
        """oof: list of (dl, tl, lab[, C]) per sequence, dl/tl out-of-fold first-stage logits, C context."""
        Xd, yd, Xt, yt = [], [], [], []
        for item in oof:
            dl, tl, lab = item[:3]
            C = item[3] if len(item) > 3 else None
            F, _ = features(dl, tl, C)
            Ff, _ = features(dl, -tl, C)
            idx = np.flatnonzero(lab >= 0)[::self.sub]
            y = lab[idx]
            Xd += [F[idx], Ff[idx]]; yd += [(y == 2).astype(int)] * 2
            live = y < 2
            Xt.append(F[idx][live]); yt.append((y[live] == 0).astype(int))
            Xt.append(Ff[idx][live]); yt.append((y[live] == 1).astype(int))     # mirrored copy
        self.md = _clf(self.hp).fit(np.concatenate(Xd), np.concatenate(yd))
        self.mt = _clf(self.hp).fit(np.concatenate(Xt), np.concatenate(yt))
        return self

    def predict(self, dl, tl, C=None):
        F, _ = features(dl, tl, C)
        Ff, _ = features(dl, -tl, C)
        return (0.5 * (_logit(self.md, F) + _logit(self.md, Ff)),
                0.5 * (_logit(self.mt, F) - _logit(self.mt, Ff)))
