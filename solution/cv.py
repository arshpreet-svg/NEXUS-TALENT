"""Cross-game validation of the decode settings, scored with the official metric on derived labels."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode, external, features, train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def fold_logits(tr, te):
    md, mt = train.fit(train.build(tr))
    P, B, team, lab = external.load(te)
    P, B = train.jitter(P, B, np.random.default_rng(7))
    X, _ = features.features(P, B, team)
    Xf, _ = features.features(P, B, (1 - team).astype(np.int8))
    dl, tl = train.predict_logits(md, mt, X, Xf)
    return dl, tl, lab


def score(dl, tl, lab, temp, lt, ld, mp, md_):
    gt = np.where(lab < 0, 3, lab).astype(np.int8)
    pred = np.full(len(gt), 3, dtype=np.int8)
    for s, e in pm.valid_runs(gt):
        pred[s:e] = decode.decode_span(dl[s:e], tl[s:e], temp, lt, ld, mp, md_)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


if __name__ == "__main__":
    folds = [fold_logits([a for a in train.GAMES if a != t], t) for t in train.GAMES]
    import pickle
    pickle.dump(folds, open(os.path.join(ROOT, "cache", "folds.pkl"), "wb"))
    grid = itertools.product([0.2, 0.4, 0.8], [3, 6, 10, 20], [6, 10, 20, 40], [25, 50])
    rows = []
    for temp, lt, ld, mp in grid:
        res = [score(*f, temp, lt, ld, mp, mp) for f in folds]
        rows.append((np.mean([r[0] for r in res]), np.mean([r[1] for r in res]),
                     [r[2] for r in res], [r[3] for r in res], temp, lt, ld, mp))
    rows.sort(key=lambda r: -r[0])
    print("spell macro  n_pred  n_gt | temp lam_team lam_dead min_len")
    for r in rows[:12]:
        print("%.3f %.3f %s %s | %s %s %s %s" % r)
