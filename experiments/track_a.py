"""Track A experiment runner: model variants on the Track C feature cache (leave-one-match-out).

    python experiments/track_a.py hp --dead '{"max_depth":4}' --team '{"max_depth":7}' --note "..."
    python experiments/track_a.py hp --sub 3 --note "denser sampling"

Same folds, seeds, decoder and metrics as solution/cv_fast.py, so numbers are directly comparable with the
log rows written by cv_fast. Every run appends a row to experiments/log.csv.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import cv_fast, dfl, train_dfl  # noqa: E402

BASE_HP = dict(max_iter=250, learning_rate=0.06, max_depth=5, min_samples_leaf=60, l2_regularization=1.0,
               random_state=0)


def _clf(hp):
    return HistGradientBoostingClassifier(**{**BASE_HP, **hp})


def _logit(m, X):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_plain(parts, dead_hp, team_hp, sub, seed=0):
    """train_dfl.fit with configurable capacity / subsampling. Returns predict(X, Xf) -> (dl, tl)."""
    Xd, yd, Xt, yt = [], [], [], []
    for X, Xf, lab, _ in parts:
        idx = np.flatnonzero(lab >= 0)[::sub]
        y = lab[idx]
        Xd.append(X[idx]); yd.append((y == 2).astype(int))
        live = y < 2
        Xt.append(X[idx][live]); yt.append((y[live] == 0).astype(int))
        yf = np.where(y == 0, 1, np.where(y == 1, 0, y))
        Xt.append(Xf[idx][live]); yt.append((yf[live] == 0).astype(int))
    md = _clf({**dead_hp, "random_state": seed}).fit(np.concatenate(Xd), np.concatenate(yd))
    mt = _clf({**team_hp, "random_state": seed}).fit(np.concatenate(Xt), np.concatenate(yt))

    def predict(X, Xf):
        return _logit(md, X), 0.5 * (_logit(mt, X) - _logit(mt, Xf))
    return predict


def run(fit_fn, seeds_train=(0,), seed_test=1, n_jobs=7):
    """Leave-one-match-out. fit_fn(list_of_train_part_lists) -> predict. Returns per-fold logits."""
    seqs, mask, ver = dfl.load_all(), train_dfl.atd_ball_mask(), cv_fast.code_version()
    tr = [cv_fast.get_parts(seqs, s, mask, ver, n_jobs) for s in seeds_train]
    te = cv_fast.get_parts(seqs, seed_test, mask, ver, n_jobs)
    matches = [q["match"] for q in seqs]
    order = sorted(set(matches))

    def fold(m):
        trp = [[p for p, mm in zip(t, matches) if mm != m] for t in tr]
        pred = fit_fn(trp)
        return [pred(X, Xf) + (lab,) for (X, Xf, lab, _), mm in zip(te, matches) if mm == m]
    return order, Parallel(n_jobs=n_jobs)(delayed(fold)(m) for m in order)


def report(order, folds, note, prm=cv_fast.DECODE, log=True, t0=None, hold=0.0):
    res = [cv_fast.fold_metrics(f, prm, hold) for f in folds]
    note = "%s [train hold 0.0s | eval hold %.1fs]" % (note, hold)
    mean = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    print("%-10s %6s %6s %6s %6s %5s" % ("fold", "spell", "macro", "tacc", "auc", "npred"))
    for m, r in zip(order, res):
        print("%-10s %6.3f %6.3f %6.3f %6.3f %5d" % (m[-6:], r["spell"], r["macro"], r["team_acc"], r["auc"], r["n_pred"]))
    print("MEAN       %6.4f %6.4f %6.4f %6.4f %5.0f   %s" % (mean["spell"], mean["macro"], mean["team_acc"], mean["auc"], mean["n_pred"], note))
    if log:
        cv_fast.append_log(dict(
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), branch=cv_fast.current_branch(), note=note, **prm,
            spellF1="%.4f" % mean["spell"], macroF1="%.4f" % mean["macro"], aliveDeadF1="%.4f" % mean["alive_dead"],
            team_acc="%.4f" % mean["team_acc"], dead_auc="%.4f" % mean["auc"], n_pred="%.0f" % mean["n_pred"],
            n_gt="%.0f" % mean["n_gt"], per_fold_spellF1=";".join("%.3f" % r["spell"] for r in res),
            seconds="%.0f" % (time.time() - t0) if t0 else ""))
    return mean, res


def stage1_nested(dh, th, sub, n_jobs=7):
    """First-stage logits for every (outer test match m, scored match j): model trained without m AND j
    (j != m, inner out-of-fold, used to train stage two) or without m only (j == m, the test logits).
    Cached on disk, keyed by the first-stage config and the feature cache version."""
    import hashlib, pickle
    seqs, mask, ver = dfl.load_all(), train_dfl.atd_ball_mask(), cv_fast.code_version()
    key = hashlib.md5(json.dumps([dh, th, sub, ver, "ctx1"], sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(ROOT, "cache", "stage1_%s.pkl" % key)
    matches = [q["match"] for q in seqs]
    order = sorted(set(matches))
    if os.path.exists(path):
        return order, matches, pickle.load(open(path, "rb"))
    tr = cv_fast.get_parts(seqs, 0, mask, ver, n_jobs)
    te = cv_fast.get_parts(seqs, 1, mask, ver, n_jobs)

    import stack

    def job(m, j):
        excl = {m, j}
        pred = fit_plain([p for p, mm in zip(tr, matches) if mm not in excl], dh, th, sub)
        return [pred(X, Xf) + (lab, stack.context(X, nm)) for (X, Xf, lab, nm), mm in zip(te, matches) if mm == j]
    jobs = [(m, j) for m in order for j in order]
    out = Parallel(n_jobs=n_jobs)(delayed(job)(m, j) for m, j in jobs)
    res = dict(zip(jobs, out))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return order, matches, res


def run_stack(dh, th, sub, s2hp, ctx=True, n_jobs=7, s2_hold=0.0):
    import stack
    order, _, res = stage1_nested(dh, th, sub, n_jobs)
    cut = 4 if ctx else 3

    def fold(m):
        oof = [s[:cut] for j in order if j != m for s in res[(m, j)]]
        if s2_hold:     # stage two learns the controlled-possession target directly
            oof = [(o[0], o[1], dfl.apply_hold(o[2], int(round(s2_hold * 25))),) + tuple(o[3:]) for o in oof]
        s2 = stack.Stage2(s2hp).fit(oof)
        return [s2.predict(*s[:2], *s[3:cut]) + (s[2],) for s in res[(m, m)]]
    stage1 = [[s[:3] for s in res[(m, m)]] for m in order]
    stage2 = Parallel(n_jobs=n_jobs)(delayed(fold)(m) for m in order)
    return order, stage1, stage2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hp", "stack"])
    ap.add_argument("--s2", default="{}", help="stage-two hyper-parameters (stack mode)")
    ap.add_argument("--dead", default="{}")
    ap.add_argument("--team", default="{}")
    ap.add_argument("--sub", type=int, default=5)
    ap.add_argument("--seeds", default="0", help="training degradation seeds, comma separated (ensemble)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--eval-holds", default="3,2,4,0", help="target holds (s); the first is the primary yardstick")
    ap.add_argument("--s2-hold", type=float, default=0.0, help="train stage two on hold-merged labels (s)")
    ap.add_argument("--no-ctx", action="store_true", help="stage two without the context features")
    ap.add_argument("--grid", action="store_true", help="decoder grid on the held-out stage-2 logits")
    for k, v in cv_fast.DECODE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    dh, th = json.loads(a.dead), json.loads(a.team)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    t0 = time.time()

    if a.mode == "stack":
        order, s1, s2 = run_stack(dh, th, a.sub, json.loads(a.s2), ctx=not a.no_ctx, s2_hold=a.s2_hold)
        prm = dict(temp=a.temp, lam_team=a.lam_team, lam_dead=a.lam_dead, min_len=a.min_len)
        holds = [float(h) for h in a.eval_holds.split(",")]
        note = a.note or "track A stack s1 dead=%s team=%s sub=%d s2=%s" % (a.dead, a.team, a.sub, a.s2)
        for h in holds:
            print("=== target hold %.1f s --- stage 1 (same config, reference)" % h)
            report(order, s1, "", prm=prm, log=False, hold=h)
            print("=== target hold %.1f s --- stage 2" % h)
            report(order, s2, note, prm=prm, log=not a.no_log, t0=t0, hold=h)
        if a.grid:
            cv_fast.grid(s2, top=15, hold_eval=holds[0])
        sys.exit(0)

    def fit_fn(trp):
        preds = [fit_plain(p, dh, th, a.sub, seed=i) for i, p in enumerate(trp)]
        if len(preds) == 1:
            return preds[0]

        def ens(X, Xf):
            out = [p(X, Xf) for p in preds]
            return np.mean([o[0] for o in out], 0), np.mean([o[1] for o in out], 0)
        return ens

    order, folds = run(fit_fn, seeds_train=seeds)
    report(order, folds, a.note or "track A hp dead=%s team=%s sub=%d seeds=%s" % (a.dead, a.team, a.sub, a.seeds),
           log=not a.no_log, t0=t0)
