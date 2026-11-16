"""Load the tactical ATD and Smart Tagging into dense numpy arrays (cached as .npz)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
import load_smart_tagging  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "tracking.npz")


def _xy(group):
    if group[0] == "N":
        return np.nan, np.nan, np.nan
    a = group.split(",")
    s = float(a[2]) if a[2][0] != "N" else np.nan
    return float(a[0]), float(a[1]), s


def parse():
    meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    length, width = GAMES[GAME]["pitch"]
    team = team_of_each_track(meta)
    k = len(team)
    P = np.full((n, k, 3), np.nan, dtype=np.float32)   # x_m, y_m, speed
    B = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            for s, g in enumerate(parts[1].split(";")):
                if g[0] != "N":
                    x, y, sp = _xy(g)
                    P[f, s] = (x * length, y * width, sp)
            if parts[2][0] != "N":
                x, y, _ = _xy(parts[2])
                B[f] = (x * length, y * width)
    return P, B, team


def load(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        z = np.load(CACHE)
        return z["P"], z["B"], z["team"]
    P, B, team = parse()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, P=P, B=B, team=team)
    return P, B, team


def tags():
    return load_smart_tagging.load()
