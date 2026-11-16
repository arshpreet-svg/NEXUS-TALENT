"""Train / validate on the DFL open Bundesliga matches (labels follow the organisers' definition).

    python solution/train_dfl.py cv      # leave-one-match-out, scored with the official spellF1
    python solution/train_dfl.py final   # train on all 7 matches -> models/gbm_dfl.joblib
"""
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, decode, dfl, features, stack  # noqa: E402
from models import Plain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "gbm_dfl.joblib")
SUB = 5
# Team-control model capacity (Track A, experiments/log.csv "A2"): deeper, more, slower, more regularised trees.
# The dead-ball model keeps the default capacity (more did not help).
TEAM_HP = dict(max_depth=8, max_iter=800, learning_rate=0.03, min_samples_leaf=200, l2_regularization=3.0)
JIT_BALL, JIT_PLAYER = 0.8, 0.3
HALVES = [(11185, 90651), (110808, 190197)]


def atd_ball_mask():
    """Real ball-visibility pattern of the provided game's ATD (unlabelled), graded frames only."""
    _, B, _ = data.load()
    obs = np.isfinite(B[:, 0])
    return np.concatenate([obs[a:b + 1] for a, b in HALVES])


def degrade(P, B, mask, rng):
    """Make precise tracking look like the ATD: ball visibility pattern, jitter, player dropouts."""
    n = len(B)
    P, B = P.copy(), B.copy()
    P += rng.normal(0, JIT_PLAYER, P.shape).astype(np.float32)
    B += rng.normal(0, JIT_BALL, B.shape).astype(np.float32)
    s = int(rng.integers(0, len(mask)))
    m = np.take(mask, np.arange(s, s + n), mode="wrap")
    B[~m] = np.nan
    k = P.shape[1]
    drop = np.repeat(rng.random((n // 50 + 1, k)) < 0.07, 50, axis=0)[:n]
    P[drop] = np.nan
    return P, B


def make(seqs, seed, mask):
    rng = np.random.default_rng(seed)
    out = []
    for q in seqs:
        P, B = degrade(q["P"], q["B"], mask, rng)
        X, names = features.features(P, B, q["team"])
        Xf, _ = features.features(P, B, (1 - q["team"]).astype(np.int8))
        out.append((X, Xf, q["lab"], names))
    return out


def fit(parts):
    Xd, yd, Xt, yt = [], [], [], []
    for X, Xf, lab, names in parts:
        idx = np.flatnonzero(lab >= 0)[::SUB]
        y = lab[idx]
        Xd.append(X[idx]); yd.append((y == 2).astype(int))
        live = y < 2
        Xt.append(X[idx][live]); yt.append((y[live] == 0).astype(int))
        yf = np.where(y == 0, 1, np.where(y == 1, 0, y))            # team-swapped copy
        Xt.append(Xf[idx][live]); yt.append((yf[live] == 0).astype(int))
    md = Plain().fit(np.concatenate(Xd), np.concatenate(yd))
    mt = Plain(TEAM_HP).fit(np.concatenate(Xt), np.concatenate(yt))
    return md, mt, names


def logits(md, mt, X, Xf):
    dl = md.logit(X)
    tl = 0.5 * (mt.logit(X) - mt.logit(Xf))
    return dl, tl


def fit_all(parts, oof_parts, match_of_part, n_jobs=7):
    """First stage on all matches, plus the second stage (solution/stack.py) trained on out-of-fold first-stage
    logits: each match is scored by a first-stage model fitted on the other matches only.
    parts / oof_parts: make() output at the training / an independent degradation seed, same sequence order."""
    from joblib import Parallel, delayed
    order = sorted(set(match_of_part))

    def oof(m):
        md, mt, _ = fit([p for p, mm in zip(parts, match_of_part) if mm != m])
        return [logits(md, mt, X, Xf) + (lab, stack.context(X, nm))
                for (X, Xf, lab, nm), mm in zip(oof_parts, match_of_part) if mm == m]
    res = Parallel(n_jobs=n_jobs)(delayed(oof)(m) for m in order)
    md, mt, names = fit(parts)
    s2 = stack.Stage2().fit([s for r in res for s in r])
    return {"dead": md, "team": mt, "stage2": s2, "names": names}


def predict_logits(model, X, Xf):
    """Final dead-ball and team logits for a feature matrix (second stage applied when the model has one)."""
    dl, tl = logits(model["dead"], model["team"], X, Xf)
    if model.get("stage2") is not None:
        dl, tl = model["stage2"].predict(dl, tl, stack.context(X, model["names"]))
    return dl, tl


def score_match(parts_logits, params):
    """Official metrics over a match's halves, concatenated with an X gap between them."""
    gt, pred = [], []
    for dl, tl, lab in parts_logits:
        g = np.where(lab < 0, 3, lab).astype(np.int8)
        p = np.full(len(g), 3, dtype=np.int8)
        for s, e in pm.valid_runs(g):
            p[s:e] = decode.decode_span(dl[s:e], tl[s:e], params["temp"], params["lam_team"],
                                        params["lam_dead"], params["min_len"], params["min_len"])
        gap = np.full(500, 3, dtype=np.int8)
        gt += [g, gap]; pred += [p, gap]
    r = pm.evaluate(np.concatenate(gt), np.concatenate(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["alive_dead_f1"]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    seqs = dfl.load_all()
    mask = atd_ball_mask()
    matches = sorted({q["match"] for q in seqs})
    print("matches", len(matches), "sequences", len(seqs),
          "| label shares H/A/D:", np.round([np.mean(np.concatenate([q["lab"][q["lab"] >= 0] for q in seqs]) == k) for k in range(3)], 3))
    if mode == "cv":
        import itertools, pickle
        folds, summ = [], []
        for m in matches:
            tr = [q for q in seqs if q["match"] != m]
            te = [q for q in seqs if q["match"] == m]
            md, mt, _ = fit(make(tr, 0, mask))
            lg = [logits(md, mt, X, Xf) + (lab,) for X, Xf, lab, _ in make(te, 1, mask)]
            folds.append(lg)
            r = score_match(lg, dict(temp=0.8, lam_team=3, lam_dead=12, min_len=25))
            live = np.concatenate([l[2] < 2 for l in lg]) & np.concatenate([l[2] >= 0 for l in lg])
            tacc = np.mean((np.concatenate([l[1] for l in lg])[live] > 0) == (np.concatenate([l[2] for l in lg])[live] == 0))
            auc = roc_auc_score(np.concatenate([l[2] == 2 for l in lg])[np.concatenate([l[2] >= 0 for l in lg])],
                                np.concatenate([l[0] for l in lg])[np.concatenate([l[2] >= 0 for l in lg])])
            print(m, "spell %.3f macro %.3f n_pred %d n_gt %d aliveDeadF1 %.3f | team acc %.3f dead AUC %.3f" % (r + (tacc, auc)), flush=True)
            summ.append(r + (tacc, auc))
        print('MEAN spell %.3f macro %.3f n_pred %.0f n_gt %.0f aliveDeadF1 %.3f | team acc %.3f dead AUC %.3f' % tuple(np.mean(summ, 0)))
        pickle.dump(folds, open(os.path.join(ROOT, "cache", "dfl_folds.pkl"), "wb"))
        rows = []
        for temp, lt, ld, ml in itertools.product([0.2, 0.4, 0.8], [3, 6, 12, 25], [6, 12, 25, 50], [25, 50, 100]):
            prm = dict(temp=temp, lam_team=lt, lam_dead=ld, min_len=ml)
            rs = np.array([score_match(f, prm) for f in folds])
            rows.append((rs[:, 0].mean(), rs[:, 1].mean(), rs[:, 2].mean(), rs[:, 3].mean(), temp, lt, ld, ml))
        rows.sort(key=lambda r: -r[0])
        print("spell macro n_pred n_gt | temp lam_team lam_dead min_len")
        for r in rows[:10]:
            print("%.3f %.3f %.0f %.0f | %s %s %s %s" % r)
    else:
        model = fit_all(make(seqs, 0, mask), make(seqs, 1, mask), [q["match"] for q in seqs])
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print("saved", MODEL_PATH)
