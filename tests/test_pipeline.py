"""Safety net for the team's changes. Run:  .venv/bin/python -m pytest tests -q
Slow checks (full reproduction) are opt-in:  RUN_SLOW=1 .venv/bin/python -m pytest tests -q
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

import decode
import features
import possession_metrics as pm
from games import half_video_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(ROOT, "submission.csv")
HALVES = [(11185, 90651), (110808, 190197)]
N_FRAMES = 199319


# ------------------------------------------------------------------ decoder
def _logits(n, dead=None, team=None):
    dl = np.full(n, -6.0)
    tl = np.zeros(n)
    if dead:
        for a, b in dead:
            dl[a:b] = 6.0
    if team:
        for a, b, v in team:
            tl[a:b] = v
    return dl, tl


def test_decoder_recovers_clean_structure():
    n = 3000
    dl, tl = _logits(n, dead=[(1000, 1500)], team=[(0, 500, 3.0), (500, 1000, -3.0), (1500, 3000, 3.0)])
    path = decode.decode_span(dl, tl, 0.8, 3, 12, 25, 25)
    runs = [(v, e - s) for s, e, v in decode.runs(path)]
    assert [v for v, _ in runs] == [decode.H_, decode.A_, decode.D_, decode.H_]
    assert abs(runs[2][1] - 500) <= 5


def test_decoder_absorbs_flicker():
    n = 2000
    rng = np.random.default_rng(0)
    tl = 0.8 * rng.standard_normal(n) + 2.0                  # weakly positive, very noisy
    dl = np.full(n, -6.0)
    path = decode.decode_span(dl, tl, 0.8, 3, 12, 25, 25)
    assert len(decode.runs(path)) <= 3                        # one possession, not dozens


def test_decoder_signature_is_stable():
    import inspect
    names = list(inspect.signature(decode.decode_span).parameters)
    assert names == ["dead_logit", "team_logit", "temp", "lam_team", "lam_dead", "min_poss", "min_dead"]


# ------------------------------------------------------------------ constants
def test_halves_match_published_boundaries():
    assert [(a, b) for _, a, b in half_video_blocks("BAR-ATH")] == HALVES
    import predict
    assert [tuple(h) for h in predict.HALVES] == HALVES


# ------------------------------------------------------------------ feature symmetry (guards Track B changes)
SYMMETRIC = [("dH1", "dA1", +1), ("dH2", "dA2", +1), ("nH3", "nA3", +1), ("nH8", "nA8", +1), ("nH15", "nA15", +1),
             ("adv", "adv", -1), ("bprog", "bprog", -1), ("d_sx", "d_sx", -1), ("d_sy", "d_sy", -1),
             ("d_cb", "d_cb", -1), ("d_tspd", "d_tspd", -1), ("d_cprog", "d_cprog", -1)]


def test_features_symmetric_under_team_swap():
    rng = np.random.default_rng(1)
    n, k = 1500, 22
    base = rng.uniform([0, 0], [105, 68], size=(k, 2))
    P = (base[None] + np.cumsum(rng.normal(0, 0.15, (n, k, 2)), axis=0)).astype(np.float32)
    B = (P[:, 3, :] + rng.normal(0, 2.0, (n, 2))).astype(np.float32)
    B[rng.random(n) < 0.3] = np.nan
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    X, names = features.features(P, B, team)
    Xf, names2 = features.features(P, B, (1 - team).astype(np.int8))
    assert names == names2
    assert X.shape[1] == len(names) and X.shape[0] == n
    ix = {m: i for i, m in enumerate(names)}
    bad = []
    for a, b, sign in SYMMETRIC:
        if a not in ix or b not in ix:
            continue
        lhs, rhs = X[:, ix[a]], sign * Xf[:, ix[b]]
        ok = np.isfinite(lhs) & np.isfinite(rhs)
        if not np.allclose(lhs[ok], rhs[ok], atol=1e-3):
            bad.append((a, b))
    assert not bad, "not symmetric under a team swap: %s" % bad


def test_new_features_must_not_use_future_labels_or_raise_on_all_nan_ball():
    P = np.random.default_rng(2).uniform(0, 100, (400, 22, 2)).astype(np.float32)
    B = np.full((400, 2), np.nan, dtype=np.float32)                # ball never seen
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    X, names = features.features(P, B, team)
    assert X.shape == (400, len(names))


# ------------------------------------------------------------------ the submission file
needs_file = pytest.mark.skipif(not os.path.exists(SUBMISSION), reason="submission.csv not present")


@needs_file
def test_submission_passes_official_validation():
    frame = pm.read_label_csv(SUBMISSION)
    assert pm.validate_submission(frame, N_FRAMES) is None


@needs_file
def test_submission_labels_only_inside_published_halves():
    frame = pm.read_label_csv(SUBMISSION)
    lab = frame["label"].astype(str).str.strip().str.upper().values
    inside = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        inside[a:b + 1] = True
    assert set(lab) <= {"H", "A", "D", "X"}
    assert (lab[~inside] == "X").all(), "labels written outside the published halves"
    assert (lab[inside] != "X").all(), "X written inside a half (costs macroF1)"


@needs_file
def test_submission_looks_like_football():
    lab = pm.read_label_csv(SUBMISSION)["label"].astype(str).values
    inside = np.concatenate([lab[a:b + 1] for a, b in HALVES])
    share = {c: float(np.mean(inside == c)) for c in "HAD"}
    assert 0.15 < share["D"] < 0.65, share
    assert 0.15 < share["H"] < 0.55 and 0.15 < share["A"] < 0.55, share
    runs = decode.runs(np.array([{"H": 0, "A": 1, "D": 2}[c] for c in inside]))
    n_poss = sum(1 for _, _, v in runs if v < 2)
    assert 150 < n_poss < 800, n_poss                       # not the 669+ of a raw baseline-like flicker nor a handful


# ------------------------------------------------------------------ slow: full reproduction
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (needs data/ and models/)")
def test_prediction_step_is_deterministic(tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / ("run%d.csv" % i)
        subprocess.check_call([sys.executable, os.path.join(ROOT, "solution", "predict.py"), "--model", "dfl", "--out", str(out)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs.append(hashlib.md5(out.read_bytes()).hexdigest())
    assert outs[0] == outs[1]
