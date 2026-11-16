"""Train the dead-ball and team-control models on other games (Metrica open sample data)."""
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import external  # noqa: E402
import features  # noqa: E402
from models import DeadModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "gbm.joblib")
GAMES = (1, 2, 3)
SUB = 5          # train on every 5th frame (neighbours are near-duplicates)
# ball jitter matched to the ATD ball-to-nearest-player distance distribution (unlabelled calibration)
JIT_BALL, JIT_PLAYER = 0.8, 0.3


def jitter(P, B, rng):
    P = P.copy()
    B = B.copy()
    P[:, :, :2] += rng.normal(0, JIT_PLAYER, P[:, :, :2].shape).astype(np.float32)
    B += rng.normal(0, JIT_BALL, B.shape).astype(np.float32)
    # player dropouts: ~7% of slot-frames lost in runs of ~1-3 s
    n, k = P.shape[:2]
    drop = rng.random((n // 50 + 1, k)) < 0.07
    drop = np.repeat(drop, 50, axis=0)[:n]
    P[drop] = np.nan
    return P, B


def build(games, seed=0):
    rng = np.random.default_rng(seed)
    out = {"Xd": [], "yd": [], "Xt": [], "yt": [], "names": None}
    for g in games:
        P, B, team, lab = external.load(g)
        P, B = jitter(P, B, rng)
        for flip in (0, 1):
            tm = team if not flip else (1 - team).astype(np.int8)
            X, names = features.features(P, B, tm)
            out["names"] = names
            idx = np.flatnonzero(lab >= 0)[::SUB]
            y = lab[idx].copy()
            if flip:
                y = np.where(y == 0, 1, np.where(y == 1, 0, y))
            if not flip:
                out["Xd"].append(X[idx]); out["yd"].append((y == 2).astype(int))
            live = y < 2
            out["Xt"].append(X[idx][live]); out["yt"].append((y[live] == 0).astype(int))
    return {k: (np.concatenate(v) if k != "names" else v) for k, v in out.items()}


def fit(d):
    md = DeadModel()
    mt = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=5,
                                        min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    md.fit(d["Xd"], d["yd"], d["names"])
    mt.fit(d["Xt"], d["yt"])
    return md, mt


def predict_logits(md, mt, X, X_flip):
    """Returns (dead_logit, team_logit[H vs A]); team logit is symmetrised over a team swap."""
    dead_logit = md.logit(X)
    p1 = mt.predict_proba(X)[:, 1].clip(1e-4, 1 - 1e-4)
    p2 = mt.predict_proba(X_flip)[:, 1].clip(1e-4, 1 - 1e-4)
    lt = 0.5 * (np.log(p1 / (1 - p1)) - np.log(p2 / (1 - p2)))
    return dead_logit, lt


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if mode == "cv":
        for te in GAMES:
            tr = [g for g in GAMES if g != te]
            md, mt = fit(build(tr))
            t = build([te], seed=1)
            pdead = md.predict_proba(t["Xd"])[:, 1]
            pt = mt.predict_proba(t["Xt"])[:, 1]
            print("train %s -> test %d | dead AUC %.3f acc %.3f | team acc (live) %.3f" % (
                tr, te, roc_auc_score(t["yd"], pdead), ((pdead > .5) == t["yd"]).mean(),
                ((pt > .5) == t["yt"]).mean()))
    else:
        d = build(list(GAMES))
        md, mt = fit(d)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"dead": md, "team": mt, "names": d["names"]}, MODEL_PATH)
        print("saved", MODEL_PATH)
