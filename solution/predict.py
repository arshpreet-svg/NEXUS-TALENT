"""Produce the submission CSV for the provided match from the trained models.

    python solution/train.py final        # trains on Metrica sample games 1+2 -> models/gbm.joblib
    python solution/predict.py --out submission.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, decode, features, tags, train, train_dfl  # noqa: E402

HALVES = [(11185, 90651), (110808, 190197)]    # published in DATA.md; everything else is X
PARAMS_METRICA = dict(temp=0.4, lam_team=3.0, lam_dead=10.0, min_len=50)   # chosen by leave-one-game-out CV (cv.py)
PARAMS_DFL = dict(temp=0.8, lam_team=3.0, lam_dead=12.0, min_len=25)        # chosen by leave-one-match-out CV (train_dfl.py cv)
PARAMS = PARAMS_DFL
MODELS = {"dfl": train_dfl.MODEL_PATH, "metrica": train.MODEL_PATH}


def decode_all(dl, tl, prm):
    labels = np.full(len(dl), "X", dtype="<U1")
    for lo, hi in HALVES:
        sl = slice(lo, hi + 1)
        path = decode.decode_span(dl[sl], tl[sl], prm["temp"], prm["lam_team"], prm["lam_dead"],
                                  prm["min_len"], prm["min_len"])
        labels[sl] = decode.LET[path]
    return labels


def predict(prm=None, use_tags=False, model="dfl"):
    prm = {**(PARAMS_DFL if model == "dfl" else PARAMS_METRICA), **(prm or {})}
    P, B, team = data.load()
    m = joblib.load(MODELS[model])
    X, names = features.features(P, B, team, segments=HALVES)
    Xf, _ = features.features(P, B, (1 - team).astype(np.int8), segments=HALVES)
    assert names == m["names"]
    if model == "dfl":
        dl, tl = train_dfl.logits(m["dead"], m["team"], X, Xf)
    else:
        dl, tl = train.predict_logits(m["dead"], m["team"], X, Xf)
    labels = decode_all(dl, tl, prm)
    if use_tags:
        # pass 2: Smart Tagging team evidence, weighted by label-free agreement with pass 1
        bias, _ = tags.tag_bias(data.tags(), labels, len(B))
        labels = decode_all(dl, tl + bias, prm)
    return labels, dl, tl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["dfl", "metrica"], default="dfl")
    ap.add_argument("--tags", action="store_true",
                    help="add Smart Tagging team evidence (tried, NOT used in the submission; see SOLUTION.md)")
    a = ap.parse_args()
    lab, _, _ = predict(use_tags=a.tags, model=a.model)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(lab))
    print("wrote", a.out)
