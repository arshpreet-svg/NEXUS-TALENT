"""Fast leave-one-match-out validation on the DFL matches (Track C harness).

    python solution/cv_fast.py --note "baseline"            # run, print table, append a row to experiments/log.csv
    python solution/cv_fast.py --note "depth 7" --no-log     # run without logging
    python solution/cv_fast.py --grid                        # also grid-search the decoder on the out-of-fold logits

Why it is fast
  * Degraded features are cached per (half-match, seed) in cache/feat/, keyed on a hash of the code that
    produces them (features*.py, dfl.py, the degrade() source and its constants). Changing a MODEL
    (train_dfl.fit / logits, models.py) reuses the cache; changing FEATURES or DEGRADATION rebuilds it.
  * Cache misses are computed in parallel, and so are the seven folds.
It uses train_dfl.fit / train_dfl.logits / train_dfl.score_match, so model changes flow in automatically.
Note: per-sequence random streams (seed, index) differ from train_dfl.py's shared stream, so numbers are not
bit-identical to `train_dfl.py cv`; use this harness consistently as the team's yardstick.
"""
import argparse
import csv
import datetime
import hashlib
import inspect
import itertools
import os
import pickle
import subprocess
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl, features, train_dfl  # noqa: E402

ROOT = os.path.dirname(HERE)
FEAT_DIR = os.path.join(ROOT, "cache", "feat")
LOG = os.path.join(ROOT, "experiments", "log.csv")
FOLDS_PKL = os.path.join(ROOT, "cache", "dfl_folds.pkl")
TRAIN_SEED, TEST_SEED = 0, 1
DECODE = dict(temp=0.4, lam_team=6, lam_dead=25, min_len=50)       # current main settings (chosen on targets of 2-4 s hold)
LOG_COLS = ["date", "branch", "note", "temp", "lam_team", "lam_dead", "min_len", "spellF1", "macroF1",
            "aliveDeadF1", "team_acc", "dead_auc", "n_pred", "n_gt", "per_fold_spellF1", "seconds"]


def code_version():
    """Hash of everything that determines the cached features (NOT the models)."""
    h = hashlib.md5()
    for f in sorted(os.listdir(HERE)):
        if f.startswith("features") and f.endswith(".py") or f == "dfl.py":
            h.update(open(os.path.join(HERE, f), "rb").read())
    h.update(inspect.getsource(train_dfl.degrade).encode())
    h.update(repr((train_dfl.JIT_BALL, train_dfl.JIT_PLAYER)).encode())
    return h.hexdigest()[:12]


def _key(i, q, seed, ver):
    return os.path.join(FEAT_DIR, "%s_%02d_%s_s%d.npz" % (ver, i, q["half"][:5], seed))


def _compute(i, q, seed, mask):
    rng = np.random.default_rng([seed, i])
    P, B = train_dfl.degrade(q["P"], q["B"], mask, rng)
    X, names = features.features(P, B, q["team"])
    Xf, _ = features.features(P, B, (1 - q["team"]).astype(np.int8))
    return X, Xf, names


def get_parts(seqs, seed, mask, ver, n_jobs):
    """Cached (X, Xf, lab, names) for every sequence at one seed; computes misses in parallel."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    need = [i for i, q in enumerate(seqs) if not os.path.exists(_key(i, q, seed, ver))]
    if need:
        print("  computing features for %d sequence(s), seed %d ..." % (len(need), seed), flush=True)
        out = Parallel(n_jobs=n_jobs)(delayed(_compute)(i, seqs[i], seed, mask) for i in need)
        for i, (X, Xf, names) in zip(need, out):
            np.savez(_key(i, seqs[i], seed, ver), X=X, Xf=Xf, names=np.array(names))
    parts = []
    for i, q in enumerate(seqs):
        z = np.load(_key(i, q, seed, ver))
        parts.append((z["X"], z["Xf"], q["lab"], list(z["names"])))
    return parts


def run_fold(m, matches_of_seq, train_parts, test_parts, hold_train=0.0):
    tr = [(X, Xf, dfl.apply_hold(lab, int(round(hold_train * 25))), nm)
          for (X, Xf, lab, nm), mm in zip(train_parts, matches_of_seq) if mm != m]
    te = [p for p, mm in zip(test_parts, matches_of_seq) if mm == m]
    md, mt, _ = train_dfl.fit(tr)
    return [train_dfl.logits(md, mt, X, Xf) + (lab,) for X, Xf, lab, _ in te]


def translate(lg, hold):
    """Same logits, labels merged to a 'controlled possession' target with the given hold (seconds)."""
    if not hold:
        return lg
    h = int(round(hold * 25))
    return [(dl, tl, dfl.apply_hold(lab, h)) for dl, tl, lab in lg]


def fold_metrics(lg, prm, hold_eval=0.0):
    lg = translate(lg, hold_eval)
    r = train_dfl.score_match(lg, prm)               # spell, macro, n_pred, n_gt, aliveDeadF1
    lab = np.concatenate([l[2] for l in lg])
    dl, tl = np.concatenate([l[0] for l in lg]), np.concatenate([l[1] for l in lg])
    known, live = lab >= 0, (lab >= 0) & (lab < 2)
    tacc = float(np.mean((tl[live] > 0) == (lab[live] == 0)))
    auc = float(roc_auc_score(lab[known] == 2, dl[known]))
    return dict(spell=r[0], macro=r[1], n_pred=r[2], n_gt=r[3], alive_dead=r[4], team_acc=tacc, auc=auc)


def current_branch():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def append_log(row):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def _grid_cell(folds, prm, hold_eval=0.0):
    rs = np.array([train_dfl.score_match(translate(f, hold_eval), prm) for f in folds])
    return (rs[:, 0].mean(), rs[:, 1].mean(), rs[:, 2].mean(), rs[:, 3].mean(), prm["temp"], prm["lam_team"], prm["lam_dead"], prm["min_len"])


def grid(folds, top=10, n_jobs=7, hold_eval=0.0):
    cells = [dict(temp=t, lam_team=lt, lam_dead=ld, min_len=ml)
             for t, lt, ld, ml in itertools.product([0.4, 0.8, 1.6], [1, 2, 3, 6], [3, 6, 12, 25], [10, 25, 50])]
    rows = Parallel(n_jobs=n_jobs)(delayed(_grid_cell)(folds, c, hold_eval) for c in cells)
    rows.sort(key=lambda r: -r[0])
    print("\ndecoder grid on out-of-fold logits (top %d of %d)\n spell  macro n_pred n_gt | temp lam_team lam_dead min_len" % (top, len(rows)))
    for r in rows[:top]:
        print(" %.3f  %.3f  %4.0f  %4.0f | %s %s %s %s" % r)
    print(" ... worst %.3f" % rows[-1][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--note", default="", help="what this run changes (goes into the log)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--grid", action="store_true", help="also grid-search the decoder")
    ap.add_argument("--jobs", type=int, default=min(7, os.cpu_count() or 1))
    ap.add_argument("--hold", type=float, default=0.0, help="TRAINING labels: merge possessions shorter than this many seconds")
    ap.add_argument("--eval-holds", type=str, default="3,0",
                    help="comma list of target holds (s) to score against; the FIRST is the primary yardstick. "
                         "3 = 'controlled possession' (touch-level flips under 3 s merged), 0 = raw DFL labels")
    for k, v in DECODE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    prm = dict(temp=a.temp, lam_team=a.lam_team, lam_dead=a.lam_dead, min_len=a.min_len)

    t0 = time.time()
    seqs, mask, ver = dfl.load_all(), train_dfl.atd_ball_mask(), code_version()
    print("features cache version %s | %d half-matches" % (ver, len(seqs)))
    tr_parts = get_parts(seqs, TRAIN_SEED, mask, ver, a.jobs)
    te_parts = get_parts(seqs, TEST_SEED, mask, ver, a.jobs)
    matches = [q["match"] for q in seqs]
    order = sorted(set(matches))
    folds = Parallel(n_jobs=a.jobs)(delayed(run_fold)(m, matches, tr_parts, te_parts, a.hold) for m in order)
    holds = [float(x) for x in a.eval_holds.split(",")] if a.eval_holds else [a.hold]
    print("training labels: hold %.1f s | scoring against target holds %s" % (a.hold, holds))
    for ev in holds[1:]:
        rr = [fold_metrics(f, prm, ev) for f in folds]
        print("  target hold %.1f s: spell %.3f macro %.3f n_pred %.0f n_gt %.0f" % (
            ev, np.mean([r["spell"] for r in rr]), np.mean([r["macro"] for r in rr]),
            np.mean([r["n_pred"] for r in rr]), np.mean([r["n_gt"] for r in rr])))
        if not a.no_log:
            append_log(dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), branch=current_branch(),
                            note="%s [train hold %.1fs | eval hold %.1fs]" % (a.note, a.hold, ev), **prm,
                            spellF1="%.4f" % np.mean([r["spell"] for r in rr]), macroF1="%.4f" % np.mean([r["macro"] for r in rr]),
                            aliveDeadF1="%.4f" % np.mean([r["alive_dead"] for r in rr]), team_acc="%.4f" % np.mean([r["team_acc"] for r in rr]),
                            dead_auc="%.4f" % np.mean([r["auc"] for r in rr]), n_pred="%.0f" % np.mean([r["n_pred"] for r in rr]),
                            n_gt="%.0f" % np.mean([r["n_gt"] for r in rr]),
                            per_fold_spellF1=";".join("%.3f" % r["spell"] for r in rr), seconds=""))
    res = [fold_metrics(f, prm, holds[0]) for f in folds]
    print("\n%-16s %7s %7s %8s %8s %8s %6s %6s" % ("held-out match", "spell", "macro", "aliveDF1", "teamAcc", "deadAUC", "n_pred", "n_gt"))
    for m, r in zip(order, res):
        print("%-16s %7.3f %7.3f %8.3f %8.3f %8.3f %6d %6d" % (m[-6:], r["spell"], r["macro"], r["alive_dead"], r["team_acc"], r["auc"], r["n_pred"], r["n_gt"]))
    mean = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    sec = time.time() - t0
    print("%-16s %7.3f %7.3f %8.3f %8.3f %8.3f %6.0f %6.0f   (%.0fs)" % (
        "MEAN", mean["spell"], mean["macro"], mean["alive_dead"], mean["team_acc"], mean["auc"], mean["n_pred"], mean["n_gt"], sec))

    os.makedirs(os.path.dirname(FOLDS_PKL), exist_ok=True)
    pickle.dump(folds, open(FOLDS_PKL, "wb"))
    if not a.no_log:
        append_log(dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), branch=current_branch(), note="%s [train hold %.1fs | eval hold %.1fs]" % (a.note, a.hold, holds[0]), **prm,
                        spellF1="%.4f" % mean["spell"], macroF1="%.4f" % mean["macro"], aliveDeadF1="%.4f" % mean["alive_dead"],
                        team_acc="%.4f" % mean["team_acc"], dead_auc="%.4f" % mean["auc"],
                        n_pred="%.0f" % mean["n_pred"], n_gt="%.0f" % mean["n_gt"],
                        per_fold_spellF1=";".join("%.3f" % r["spell"] for r in res), seconds="%.0f" % sec))
        print("logged to", os.path.relpath(LOG, ROOT))
    if a.grid:
        grid(folds, n_jobs=a.jobs, hold_eval=holds[0])


if __name__ == "__main__":
    main()
