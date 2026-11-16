"""Metrica Sports open sample games 1 and 2 (other matches, public): tracking + events.

Possession and dead-ball labels are derived from the human-coded events of THESE games only
(allowed by RULES.md: hand labels on other games). Nothing here touches the provided game.
Source: https://github.com/metrica-sports/sample-data (anonymised sample data).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
L, W = 105.3, 68.0
H_, A_, D_ = 0, 1, 2
POSS_TYPES = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "CARRY"}


def _read(path):
    df = pd.read_csv(path, skiprows=3, header=None)
    return df


def load_tracking(g):
    h = _read(os.path.join(EXT, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % g))
    a = _read(os.path.join(EXT, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % g))
    n = len(h)
    hv, av = h.iloc[:, 3:-2].values.astype(np.float64), a.iloc[:, 3:-2].values.astype(np.float64)
    ball = h.iloc[:, -2:].values.astype(np.float64)
    ph = hv.reshape(n, -1, 2)
    pa = av.reshape(n, -1, 2)
    P = np.concatenate([ph, pa], axis=1)
    P[:, :, 0] *= L
    P[:, :, 1] *= W
    ball[:, 0] *= L
    ball[:, 1] *= W
    team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
    return P.astype(np.float32), ball.astype(np.float32), team


def load_events(g):
    if g != 3:
        return pd.read_csv(os.path.join(EXT, "Sample_Game_%d_RawEventsData.csv" % g))
    import json
    rows = []
    for ev in json.load(open(os.path.join(EXT, "Sample_Game_3_events.json")))["data"]:
        st = ev["subtypes"]
        names = [x["name"] for x in st] if isinstance(st, list) else ([st["name"]] if st else [])
        rows.append({"Team": "Away" if ev["team"]["id"] == "FIFATMB" else "Home",
                     "Type": ev["type"]["name"], "Subtype": "-".join(names) or np.nan,
                     "Period": ev["period"], "Start Frame": ev["start"]["frame"],
                     "End Frame": ev["end"]["frame"]})
    return pd.DataFrame(rows)


def load_tracking_3():
    rows = []
    with open(os.path.join(EXT, "Sample_Game_3_tracking.txt")) as f:
        for line in f:
            p = line.split(":")
            rows.append([float(v) for g in p[1].split(";") for v in g.split(",")]
                        + [float(v) for v in p[2].split(",")])
    T = np.array(rows, dtype=np.float64)
    P = T[:, :44].reshape(-1, 22, 2).copy()
    B = T[:, 44:46].copy()
    P[:, :, 0] *= L; P[:, :, 1] *= W; B[:, 0] *= L; B[:, 1] *= W
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)   # slots 0-10 team A, 11-21 team B (verified vs events)
    return P.astype(np.float32), B.astype(np.float32), team


def derive_labels(g, n):
    """Per-frame code: 0 home possession, 1 away, 2 dead, -1 unlabelled."""
    e = load_events(g)
    e = e.sort_values("Start Frame", kind="stable").reset_index(drop=True)
    e["t"] = (e["Start Frame"] - 1).clip(lower=0)
    lab = np.full(n, -1, dtype=np.int8)
    sub = e["Subtype"].fillna("").str.upper()
    # live possession: team of the latest possession event
    pe = e[e.Type.isin(POSS_TYPES)].copy()
    pe["code"] = (pe.Team == "Away").astype(int)
    starts, codes = pe["t"].values, pe["code"].values
    for k in range(len(starts)):
        end = starts[k + 1] if k + 1 < len(starts) else min(n, starts[k] + 100)
        lab[starts[k]:min(end, n)] = codes[k]
    # dead: from a stoppage event to the next set-piece restart
    sp = e[e.Type == "SET PIECE"]["t"].values
    ender = (e.Type.isin(["BALL OUT", "FAULT RECEIVED"])
             | ((e.Type == "SHOT") & sub.str.contains("GOAL"))
             | ((e.Type == "BALL LOST") & sub.str.contains("OFFSIDE|HAND BALL")))
    for t in e[ender]["t"].values:
        nxt = sp[sp > t]
        if len(nxt) and nxt[0] - t < 25 * 120:
            lab[t:nxt[0]] = D_
    # keep only frames inside each period's event span
    keep = np.zeros(n, bool)
    for _, grp in e.groupby("Period"):
        keep[int(grp["t"].min()):min(n, int(grp["End Frame"].max()))] = True
    lab[~keep] = -1
    return lab


def load(g):
    P, B, team = load_tracking_3() if g == 3 else load_tracking(g)
    return P, B, team, derive_labels(g, len(B))


if __name__ == "__main__":
    for g in (1, 2, 3):
        P, B, team, lab = load(g)
        v = lab[lab >= 0]
        print(g, P.shape, "labelled", len(v), "H %.2f A %.2f D %.2f" % tuple((v == k).mean() for k in range(3)),
              "ball obs %.3f" % np.isfinite(B[:, 0]).mean())
