"""Dead-ball model that cannot use 'the ball is untracked' as a shortcut.

In the Metrica training games the ball is almost never tracked while it is out of play, so a model
that sees ball visibility learns 'ball missing => dead, ball visible => live'. That is false for the
provided game's tracking (the ball is often visible during stoppages). Two models avoid it:
  * ball model  : trained ONLY on frames whose surrounding window has the ball visible, visibility
                  features removed -> it must tell dead from live from geometry/motion alone;
  * player model: only collective player-motion features (nothing about the ball) on all frames.
At inference they are blended by how visible the ball is around the frame.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

VIS = "obs_m6"
LEAKY = ("obs", "since_ball", "until_ball")       # prefixes of ball-visibility features
PLAYER_ONLY = ("pspd", "pstill", "nplay")
LO, HI = 0.25, 0.75                               # blend ramp on window ball visibility


def _hgb(**hp):
    kw = dict(max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=60, l2_regularization=1.0,
              random_state=0)
    return HistGradientBoostingClassifier(**{**kw, **hp})


class DeadModel:
    def fit(self, X, y, names):
        self.names = list(names)
        self.vis = self.names.index(VIS)
        self.ball_cols = [i for i, n in enumerate(self.names) if not n.startswith(LEAKY)]
        self.player_cols = [i for i, n in enumerate(self.names) if n.startswith(PLAYER_ONLY)]
        seen = X[:, self.vis] >= 0.5
        self.ball = _hgb().fit(X[seen][:, self.ball_cols], y[seen])
        self.player = _hgb().fit(X[:, self.player_cols], y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def logit(self, X):
        a = self._logit(self.ball.predict_proba(X[:, self.ball_cols])[:, 1])
        b = self._logit(self.player.predict_proba(X[:, self.player_cols])[:, 1])
        w = np.clip((np.nan_to_num(X[:, self.vis], nan=0.0) - LO) / (HI - LO), 0, 1)
        return w * a + (1 - w) * b

    def predict_proba(self, X):                    # sklearn-style, used by the CV printout
        p = 1 / (1 + np.exp(-self.logit(X)))
        return np.stack([1 - p, p], axis=1)


class Plain:
    """Single gradient-boosted classifier with a logit() interface (used when labels are unbiased)."""

    def __init__(self, hp=None):
        self.hp = hp or {}

    def fit(self, X, y, names=None):
        self.model = _hgb(**self.hp).fit(X, y)
        return self

    def logit(self, X):
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        return self.model.predict_proba(X)
