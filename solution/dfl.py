"""DFL / Sportec open Bundesliga data (Bassek et al., Sci Data 2025, CC BY 4.0, figshare 28196177).

7 matches, 25 fps tracking with, per frame, the ball status (alive/dead) and the team in possession,
i.e. the same definition the organisers describe for their ground truth. Used only as TRAINING and
VALIDATION data from OTHER matches (allowed by RULES.md). Each half is one training sequence.
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "external", "dfl")
CACHE = os.path.join(ROOT, "cache", "dfl")
H_, A_, D_ = 0, 1, 2


def match_ids():
    out = []
    for f in sorted(os.listdir(DIR)):
        m = re.match(r"DFL_04_03_positions_raw_observed_(DFL-COM-\d+)_(DFL-MAT-\w+)\.xml", f)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _teams(mid):
    f = [x for x in os.listdir(DIR) if x.startswith("DFL_02_01_matchinformation") and mid in x][0]
    g = ET.parse(os.path.join(DIR, f)).getroot().find(".//General")
    return g.get("HomeTeamId"), g.get("GuestTeamId")


def parse(com, mid):
    """Returns {half: dict(P, B, team, status, poss)} with P (n, slots, 2) in 0..105 x 0..68 metres."""
    home, guest = _teams(mid)
    path = os.path.join(DIR, "DFL_04_03_positions_raw_observed_%s_%s.xml" % (com, mid))
    sets = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "FrameSet":
            continue
        half, team, pid = el.get("GameSection"), el.get("TeamId"), el.get("PersonId")
        if team != "referee":
            fr = el.findall("Frame")
            N = np.array([int(f.get("N")) for f in fr])
            X = np.array([float(f.get("X")) for f in fr])
            Y = np.array([float(f.get("Y")) for f in fr])
            extra = None
            if team not in (home, guest):                    # the ball
                extra = (np.array([int(f.get("BallStatus", -1)) for f in fr]),
                         np.array([int(f.get("BallPossession", -1)) for f in fr]))
            sets.setdefault(half, []).append((team, pid, N, X, Y, extra))
        el.clear()
    out = {}
    for half, items in sets.items():
        n0 = min(it[2].min() for it in items)
        n1 = max(it[2].max() for it in items)
        n = n1 - n0 + 1
        players = [it for it in items if it[5] is None]
        ball = [it for it in items if it[5] is not None]
        if not ball:
            continue
        P = np.full((n, len(players), 2), np.nan, dtype=np.float32)
        team = np.zeros(len(players), dtype=np.int8)
        for k, (tm, pid, N, X, Y, _) in enumerate(players):
            P[N - n0, k, 0], P[N - n0, k, 1] = X + 52.5, Y + 34.0
            team[k] = 0 if tm == home else 1
        _, _, N, X, Y, (st, po) = ball[0]
        B = np.full((n, 2), np.nan, dtype=np.float32)
        B[N - n0, 0], B[N - n0, 1] = X + 52.5, Y + 34.0
        S = np.full(n, -1, dtype=np.int8); S[N - n0] = st
        Pp = np.full(n, -1, dtype=np.int8); Pp[N - n0] = po
        out[half] = dict(P=P, B=B, team=team, status=S, poss=Pp)
    return out


def labels(d):
    """0 home / 1 away possession, 2 dead, -1 unknown -- from the tracking's own ball status + possession."""
    S, Pp = d["status"], d["poss"]
    lab = np.full(len(S), -1, dtype=np.int8)
    lab[(S == 1) & (Pp == 1)] = H_
    lab[(S == 1) & (Pp == 2)] = A_
    lab[S == 0] = D_
    # grade only the playing span of the half (first to last live frame), like the organisers
    alive = np.flatnonzero(S == 1)
    if len(alive):
        lab[:alive[0]] = -1
        lab[alive[-1] + 1:] = -1
    return lab


def load_all(refresh=False):
    """List of sequences (one per half): dict(P, B, team, lab, match, half)."""
    os.makedirs(CACHE, exist_ok=True)
    seqs = []
    for com, mid in match_ids():
        f = os.path.join(CACHE, mid + ".npz")
        if os.path.exists(f) and not refresh:
            z = np.load(f, allow_pickle=True)
            seqs.extend(z["seqs"].tolist())
            continue
        halves = parse(com, mid)
        these = [dict(P=d["P"], B=d["B"], team=d["team"], lab=labels(d), match=mid, half=h)
                 for h, d in sorted(halves.items())]
        np.savez_compressed(f, seqs=np.array(these, dtype=object))
        seqs.extend(these)
    return seqs
