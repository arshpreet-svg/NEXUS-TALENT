# Technical document — possession labelling, BAR-ATH (FC Barcelona vs Athletic Club)

## Summary
Per-frame possession labels (`H` / `A` / `D` / `X`) for the tactical video, produced by a learned
two-model system plus a sequence decoder that uses **only the automatic tracking (ATD)** of the tactical
video. The models are trained entirely on **other matches** (seven Bundesliga matches from the public DFL
open dataset), after degrading that precise tracking so that it looks like ours. No label, correction or
constant comes from the provided match.

| | |
|---|---|
| Training data | DFL open Bundesliga data, 7 matches (14 halves), CC BY 4.0 |
| Models | gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) |
| Decoder | 3-state Viterbi over H / A / D |
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
2. **Dead-ball model** and **team-control model** (`solution/train_dfl.py`, `solution/models.py`): gradient-boosted
   trees (250 iterations, learning rate 0.06, depth 5, minimum leaf 60, L2 1.0), trained on every fifth frame.
   They output P(dead) and P(home | ball alive) per frame. The team model is trained on each frame in both
   team orientations and its prediction is symmetrised, `½[logit(x) − logit(x with the teams swapped)]`, so it
   has no home bias.
3. **Decoder** (`solution/decode.py`): a 3-state Viterbi over the log-probabilities (emission temperature
   0.8, switch cost 3 for H↔A and 12 to/from D), then runs shorter than 1 s (25 frames) are absorbed by their
   neighbours. This is what turns noisy per-frame evidence into possessions with realistic lengths: the
   nearest-player baseline invents 669 possessions, this system produces 297.
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

## Validation (the provided game's labels are never used for training or selection)
* **Leave-one-match-out**: train on six matches, test on the seventh (degraded with a different random seed),
  seven folds, scored with the official `evaluation/possession_metrics.py`. All model and decoder choices were
  made on this validation (`solution/cv_fast.py`, one row per experiment in `experiments/log.csv`).

  | mean of 7 held-out matches | value |
  |---|---|
  | spellF1 | 0.468 |
  | macroF1 | 0.858 |
  | alive / dead F1 | 0.952 |
  | dead-ball AUC | 0.989 |
  | team accuracy on live frames | 0.857 |
  | predicted vs true possessions per match | 262 vs 437 |

  Progression on the same validation: base features 0.369 → + team-shape features 0.444 → + field-position
  features about 0.46–0.47 spellF1 (each step improved every fold).
* **Decoder sensitivity**: a 144-cell grid over temperature, switch costs and minimum run gives 0.442–0.470; the
  chosen setting (0.468) is within 0.002 of the best, so it is not sensitive.
* **Public 5-minute window** — used only to catch alignment / orientation bugs and, early on, to test two
  assumptions (see "Tried and dropped"); no model or parameter was selected on it. For the submitted file:
  macroF1 0.865, spellF1 0.516, 14 predicted vs 17 true possessions, 77 % of true dead frames recovered
  (floor baseline on the same window: 0.621 / 0.326). With 17 possessions one possession moves spellF1 by about
  0.06, so this figure ranks nothing (EVALUATION.md).
* **Full match, no labels needed**: H 53,457 / A 32,305 / D 73,095 graded frames (D = 46 %, in line with real effective
  playing time and the 44.5 % in the training matches); 297 possessions of mean 11.6 s.

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
    vs 0.468 for Viterbi — no gain, unused.
* **Not used**: the videos, the homography, the broadcast recording.

## Possession rules implemented
Possession is the team the control model assigns the ball to, held by the decoder until sustained evidence (the
switch cost and the 1 s minimum run) says otherwise. Loose balls, aerial duels and deflections therefore stay with
the previous team; a hand-over needs several consecutive frames of evidence. Dead ball follows the dead-ball model
(slow collective movement, ball at or over the edge, ball unseen for long, restart formations). Restarts are not
handled by a special rule; a possession begins when the decoder leaves a dead stretch.

## Limitations
* **Short possessions are not recovered.** In the DFL matches 44 % of possessions last under 3 s (touch-level
  flips) and only 1–14 % of them are matched; possessions over 15 s are matched 88 % of the time. DFL's possession
  definition is finer than a "controlled possession" reading, so validation spellF1 on DFL has a low ceiling that is
  partly a label-granularity artefact. The organisers' ground truth may differ in granularity from DFL's.
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
* Training takes a few minutes; prediction a few seconds. The pipeline is deterministic: two from-scratch runs
  (caches and models deleted) gave a byte-identical file (MD5 `50d61d83ee15074347884f1d2058f35a`). Python 3.15.0,
  pinned versions in `requirements-solution.txt`.
* Validation: `python solution/cv_fast.py --note "..."` (about 30 s with a warm cache). Tests:
  `pip install -r requirements-dev.txt && python -m pytest tests -q`.
* The repository contains no match data; `data/`, `cache/` and `models/` are gitignored.
