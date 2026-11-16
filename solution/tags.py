"""Smart Tagging as weak, label-free evidence about which team has the ball.

Each tagged phase carries a Team. Its meaning depends on the code:
  * 'DEFENDING IN ...' and 'DEFENSIVE TRANSITION' name the DEFENDING team -> possession is the other side
    (also what the code names say; confirmed by the model agreeing with them only 12-32 % un-inverted).
  * 'GOALS' is ambiguous and skipped; 'N/A' teams are skipped.
Reliability of each code is estimated WITHOUT any game labels: the tag and the (independently built)
tracking model are two noisy annotators, so
        agree = a_model * a_tag + (1 - a_model) * (1 - a_tag)
and a_tag follows from the measured agreement and the model's cross-game accuracy (a_model, from
solution/train.py cv). The tag then adds logit(a_tag) to the team log-odds on the frames it covers.
The estimate is capped at 0.9 because both signals derive from the same video (errors may correlate).
"""
import numpy as np

CLUB = {"Barcelona": +1, "Atletic Club": -1}     # +1 = home (H), -1 = away (A)
A_MODEL = 0.87                                    # live-frame team accuracy on held-out games
CAP = 0.9
MIN_A = 0.55
CLIP = 1.5


def _sign(event):
    team = CLUB.get(event["tags"].get("Team"))
    code = (event["code"] or "").upper()
    if team is None or code == "GOALS":
        return 0
    return -team if (code.startswith("DEFENDING") or code == "DEFENSIVE TRANSITION") else team


def tag_bias(events, labels, n):
    """Returns (bias[n] added to the home-vs-away logit, per-code report)."""
    live = (labels == "H") | (labels == "A")
    agree = {}
    for e in events:
        s = _sign(e)
        if not s:
            continue
        seg = labels[e["start_frame"]:e["end_frame"]]
        m = (seg == "H") | (seg == "A")
        a = agree.setdefault(e["code"], [0, 0])
        a[0] += int((seg[m] == ("H" if s > 0 else "A")).sum())
        a[1] += int(m.sum())
    weight, report = {}, {}
    for code, (k, tot) in agree.items():
        p = k / max(tot, 1)
        a_t = (p - (1 - A_MODEL)) / (2 * A_MODEL - 1)
        a_t = float(np.clip(a_t, 0.5, CAP))
        weight[code] = 0.0 if a_t < MIN_A else float(np.log(a_t / (1 - a_t)))
        report[code] = (round(p, 3), round(a_t, 3), round(weight[code], 2))
    bias = np.zeros(n)
    for e in events:
        s = _sign(e)
        if s:
            bias[e["start_frame"]:e["end_frame"]] += s * weight[e["code"]]
    return np.clip(bias, -CLIP, CLIP), report
