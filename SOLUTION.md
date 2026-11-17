# Technical document — possession labelling, BAR-ATH (FC Barcelona vs Athletic Club)

## Summary
Per-frame possession labels (`H` / `A` / `D` / `X`) for the tactical video, produced by a learned
two-stage system plus a sequence decoder that uses **only the automatic tracking (ATD)** of the tactical
video. The models are trained entirely on **other matches** (seven Bundesliga matches from the public DFL
open dataset), after degrading that precise tracking so that it looks like ours. No label, correction or
constant comes from the provided match.

| | |
|---|---|
| Training data | DFL open Bundesliga data, 7 matches (14 halves), CC BY 4.0 |
| Models | gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`): per-frame stage one, temporal stage two |
| Decoder | 3-state Viterbi over H / A / D (smooth: two-second minimum run) |
| Held-out validation | leave-one-match-out over the 7 matches, scored with the official metric code |
| Output | 199,319 rows; labels only inside the two published halves, `X` elsewhere |

## Approach

1. **Features** (`solution/features.py`, 97 per frame, geometry only, no player identities).
   Computed on positions in metres; ball gaps up to 1 s are linearly interpolated.
   * *Ball–player geometry*: distance from the ball to the nearest and second-nearest player of each team,
     the advantage between the two teams, player counts within 3 / 8 / 15 m, distance of the ball to the
     pitch edge, ball speed.
   * *Collective motion*: median player speed (robust to the tracker's identity switches), fraction of
     standing players, number of tracked players.
   * *Ball visibility*: visibility ratio, time since / until the ball was last seen.
   * *Team shape* (orientation-free): spread of each team in x and y, distance of each team's centroid to
     the ball, mean team speed, pressure on the player nearest the ball, that player's speed and team.
   * *Field position*: each team's attacking direction is inferred per half from where its players stand
     on average; features give the ball's progress toward each goal and the height of each team's lines.
   * Most features are also taken at 1 / 3 / 6 / 15 s centred windows.
   All features are symmetric under swapping the two teams (checked by a unit test).
2. **Stage one: dead-ball model** and **team-control model** (`solution/train_dfl.py`, `solution/models.py`):
   gradient-boosted trees trained on every fifth frame (dead-ball: 250 iterations, learning rate 0.06, depth 5,
   minimum leaf 60, L2 1.0; team-control, larger: 800 iterations, learning rate 0.03, depth 8, minimum leaf 200, L2 3.0).
   They output P(dead) and P(home | ball alive) per frame. The team model is trained on each frame in both
   team orientations and its prediction is symmetrised, `½[logit(x) − logit(x with the teams swapped)]`, so it
   has no home bias.
   **Stage two** (`solution/stack.py`): a second pair of smaller trees re-predicts both quantities from the stage-one
   logits summarised over ±1, 3, 6 and 15 s (centred, past and future means, slope, extrema) plus nine team-neutral
   context features (ball visibility, time since / until the ball was seen, player motion). It is team-swap symmetric
   like stage one and is trained **only on out-of-fold stage-one logits** (each training match scored by a stage-one
   model that never saw it), with the 3 s controlled-possession labels (see below). It is applied to each half
   separately so its windows never reach across a half boundary.
3. **Decoder** (`solution/decode.py`): a 3-state Viterbi over the log-probabilities (emission temperature
   0.4, switch cost 6 for H↔A and 25 to/from D), then runs shorter than 2 s (50 frames) are absorbed by their
   neighbours. This is what turns noisy per-frame evidence into *controlled* possessions: the nearest-player
   baseline invents 669 possessions, this system produces 199. The strength of this smoothing encodes an
   assumption about the target — see "Possession definition" below.
4. **Scope** (`solution/predict.py`): only the half boundaries published in DATA.md (frames 11185–90651 and
   110808–190197) are labelled; every other frame is `X`. No `X` is written inside a half.

### Making other matches look like ours
The DFL tracking is precise and always has the ball, while our ATD has the ball on only 57 % of frames and
is noisier. Before computing features, each training half is degraded (`train_dfl.degrade`):
* **Ball visibility**: the real ball-visible / missing pattern of the provided game's *unlabelled* ATD
  (graded frames) is tiled onto the training ball, from a random offset.
* **Jitter**: Gaussian noise of 0.8 m on the ball and 0.3 m on players. These were set by matching the
  distribution of the ball-to-nearest-player distance to our unlabelled ATD (10th percentile 0.98 m in our
  data vs 0.15 m in clean play) — a distribution match, no labels.
* **Player drop-outs**: about 7 % of player-frames lost, in 2-second blocks.

## Training data
All labels come from **other matches**; none from the provided game.
* **DFL open Bundesliga dataset** — Bassek, M., Rein, R., Weber, H., Memmert, D. (2025), *An integrated dataset of
  spatiotemporal and event data in elite soccer*, Scientific Data; figshare doi 10.6084/m9.figshare.28196177;
  licence CC BY 4.0. Seven full matches, 25 fps positions (about 2.6 GB). Per frame the provider's tracking carries
  `BallStatus` (alive / dead) and `BallPossession` (home / away); we use those fields directly: `D` where the ball is
  dead, otherwise the team named as in possession. Each half is trimmed to its playing span (first to last live
  frame). The labels were produced by the data provider, not by us; no one on the team labelled anything by hand.
  Label shares: H 29.5 %, A 26.0 %, D 44.5 %.
* `solution/download_dfl.sh` downloads the files and verifies every MD5 against the figshare manifest.

## Possession definition and the hold study
DFL flips its possession flag on **any touch**, so 44 % of its possessions last under 3 s and 437 occur per match.
CHALLENGE.md asks for *controlled* possession: "a simple touch … a deflection … an aerial duel" do not count, and a
proxy is "a player keeps the ball for a few seconds". We therefore score validation against DFL labels in which
H/A runs shorter than a **hold** are merged into their live neighbour (never into a dead stretch;
`dfl.apply_hold`), and we treat the hold as an unknown in the range 2–4 s rather than a tuned constant:

| spellF1, decoder 0.4 / 6 / 25 / 50, trained on raw DFL labels | target hold 2 s | 3 s | 4 s | raw (0 s) |
|---|---|---|---|---|
| mean of 7 held-out matches | 0.593 | **0.641** | 0.645 | 0.442 |
| previous decoder (0.8 / 3 / 12 / 25) | 0.575 | 0.592 | 0.581 | 0.468 |
| **submitted: two-stage model, decoder 0.4 / 6 / 25 / 50** | 0.593 | **0.651** | 0.657 | 0.434 |

* The smoother decoder wins on every target of 2 s or coarser and loses 0.026 on the raw touch-level target, which the
  rules text excludes.
* Training on hold-merged labels did **not** help once the decoder was re-tuned: mean spellF1 over the 2/3/4 s
  targets was 0.626 (raw labels), 0.616 (2 s), 0.615 (3 s). We therefore train on the raw DFL labels.
* The 3 s target is our primary yardstick (`solution/cv_fast.py`, default); mean possession length at a 3 s hold is
  13.8 s, consistent with the organisers' published 14.2 s, which was **not** used to choose it.

## Validation (the provided game's labels are never used for training or selection)
* **Leave-one-match-out**: train on six matches, test on the seventh (degraded with a different random seed),
  seven folds, scored with the official `evaluation/possession_metrics.py` against the 3 s controlled-possession
  target. All model and decoder choices were made on this validation (`solution/cv_fast.py`; one row per experiment
  in `experiments/log.csv`).

  | mean of 7 held-out matches | 3 s target | raw DFL labels |
  |---|---|---|
  | spellF1 | 0.651 | 0.434 |
  | macroF1 | 0.868 | 0.858 |
  | alive / dead F1 | 0.953 | 0.953 |
  | dead-ball AUC | 0.992 | 0.992 |
  | team accuracy on live frames | 0.870 | 0.870 |
  | predicted vs true possessions per match | 199 vs 233 | 199 vs 437 |

  The second stage was validated with a nested leave-one-match-out (for each held-out match, stage two is trained on
  stage-one logits from models that saw neither that match nor the match being scored). Against the single-stage model
  (3 s target 0.641) it gains 0.010; it is neutral at 2 s, +0.012 at 4 s and −0.008 on the raw target, and 2 of the 7
  folds were slightly worse, so the gain is modest. Stage two is trained on the 3 s target; training it on 2 s labels
  gave the same 3 s score (0.651).

  Progression of the model (raw-label yardstick, same decoder): base features 0.369 → + team-shape features 0.444 →
  + field-position features about 0.47 spellF1 (each step improved every fold).
* **Decoder sensitivity** (single-stage model): a 144-cell grid on the 3 s target ranges 0.55–0.64; neighbouring settings
  (team cost 6–10, dead cost 12–25, run 1–3 s) are within 0.01, so the choice is not sensitive.
* **Public 5-minute window** — used only to catch alignment / orientation bugs and, early on, to test two
  assumptions (see "Tried and dropped"); no model or parameter was selected on it. For the submitted file:
  macroF1 0.899, spellF1 0.657, 11 predicted vs 17 true possessions
  (floor baseline on the same window: 0.621 / 0.326). With 17 possessions one possession moves spellF1 by about
  0.06, so this figure ranks nothing (EVALUATION.md). The under-count (11 vs 17) is a warning that the real labels
  keep some very short possessions that this smoother decoder merges.
* **Full match, no labels needed**: H 52,879 / A 32,430 / D 73,548 graded frames (D = 46 %, in line with real effective
  playing time and the 44.5 % in the training matches); 199 possessions of mean 17.2 s.

## Assets used and dropped
* **Used**: tactical ATD (`*_atd.txt`; `*_atd.xml` only to know each track slot's team).
* **Tried and dropped**
  * *Smart Tagging as team evidence* (`solution/tags.py`, off by default). For `DEFENDING …` and `DEFENSIVE
    TRANSITION` phases the tag's team is the *defender*, so possession is the other side; we weighted each tag type by its
    label-free agreement with the model. Checking the assumption against the public window showed the tags
    agree with real possession only 65.5 % of the time (several types ≈ 50 %) — the tags are not independent of
    the model and a tagged phase is not a possession. Dropped.
  * *Metrica open sample games 1–3* (https://github.com/metrica-sports/sample-data; early models,
    `solution/external.py`, `train.py`). Labels there are derived from event data and approximate the target
    definition loosely; in those games the ball is almost never tracked while dead, which taught a model "ball
    missing ⇒ dead" (false for our data). Replaced by DFL. **Not used in the submitted file.**
  * *EM prior-shift correction* for the dead-ball model: it estimated 3 % dead among ball-visible frames, contradicted by
    the public window; removed.
  * *Hidden semi-Markov decoder* with durations fitted on other matches' labels (`decode.decode_span_hsmm`): 0.467
    vs 0.468 for Viterbi (raw-label yardstick) — no gain, unused.
  * *Training on hold-merged labels* (see the hold study): no gain once the decoder was re-tuned.
* **Not used**: the videos, the homography, the broadcast recording.
* **Other participants' public repositories** were read for ideas (how the ground truth behaves, label granularity).
  No code, model, prediction or label from them is used anywhere in this pipeline, and every number above was
  measured by us on DFL data.

## Possession rules implemented
Possession is the team the control model assigns the ball to, held by the decoder until sustained evidence (the
switch cost and the 2 s minimum run) says otherwise. Loose balls, aerial duels and deflections therefore stay with
the previous team; a hand-over needs several consecutive frames of evidence. Dead ball follows the dead-ball model
(slow collective movement, ball at or over the edge, ball unseen for long, restart formations). Restarts are not
handled by a special rule; a possession begins when the decoder leaves a dead stretch.

## Limitations
* **Short possessions are merged away.** In the DFL matches 44 % of possessions last under 3 s (touch-level flips)
  and almost none are recovered; possessions over 15 s are matched 88 % of the time. Our decoder is tuned for a
  controlled-possession target of 2–4 s; if the organisers' labels keep many shorter possessions (the public window
  has a few under 1.5 s, and we predict 11 of its 17 possessions) this costs recall. On a raw touch-level target the
  same decoder scores lower than the previous one (0.434 vs 0.468).
* **Team confusion**: home and away are confused on about 13 % of live frames; this is the largest remaining error,
  concentrated in stretches without the ball.
* **Label-definition gap**: DFL's flags follow the same two-field convention the organisers describe, but we could
  only verify agreement on the 5-minute public window.
* **Domain gap**: DFL is precise tracking from another league; our degradation matches ball visibility and
  the ball-to-player distance distribution, not every property of the ATD (e.g. identity switches).
* With more time: a learned sequence model on top of the frame models, ball estimation where it is missing, more open
  training matches, video-based stoppage cues.

## Reproduction
```
python3 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
# place the provided bundle in data/ (see DATA.md), then:
./make_submission.sh submission.csv      # downloads DFL (verified), trains, predicts, validates
```
* Training takes about 10 minutes (the nested stage-two fits dominate); prediction a few seconds. The CSV is written with
  `\n` line endings on every OS. The pipeline is deterministic: two from-scratch runs
  (caches and models deleted) gave a byte-identical file (MD5 `ec1c254b794f7b0cbc14565df02a3927`). Python 3.15.0,
  pinned versions in `requirements-solution.txt`.
* Validation: `python solution/cv_fast.py --note "..."` (about 30 s with a warm cache). Tests:
  `pip install -r requirements-dev.txt && python -m pytest tests -q`.
* The repository contains no match data; `data/`, `cache/` and `models/` are gitignored.
