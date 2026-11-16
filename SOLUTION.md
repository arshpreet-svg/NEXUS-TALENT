# Technical document — possession labelling, BAR-ATH

## Approach
A learned two-model system plus a sequence decoder, using only the tracking (ATD) of the tactical
video. Everything is trained on **other matches**; nothing is labelled or tuned on the provided game.

1. **Features** (`solution/features.py`): per frame, geometry only — distances from the ball to the
   nearest / second-nearest player of each team, player counts within 3/8/15 m, ball speed,
   distance of the ball to the pitch edge, collective player speed (median over tracks, robust to
   identity switches), fraction of standing players, ball-visibility ratios and time since/until the
   ball was last seen, each at 1/3/6/15 s centred windows. Ball gaps up to 1 s are interpolated.
   No player identity is used. The same code runs on any tracking source.
2. **Team-control model** (`solution/train.py`): gradient-boosted trees (scikit-learn
   `HistGradientBoostingClassifier`) predicting P(home | ball alive). Trained on both team
   orientations and symmetrised (`½[logit(x) − logit(x with teams swapped)]`), so it has no home bias.
3. **Dead-ball model** (`solution/models.py`), built to avoid a data artifact. In the training games
   the ball is almost never tracked while out of play, so a naive model learns "ball missing ⇒ dead",
   which is false for this game (the ball is often visible during stoppages). Two models are blended
   by local ball visibility: a *ball model* trained only on frames whose window has the ball visible
   (visibility features removed) and a *player-motion model* using only collective player motion.
4. **Decoder** (`solution/decode.py`): 3-state (H/A/D) Viterbi over the log-probabilities (emission
   temperature 0.4, switch cost 3 for H↔A and 10 to/from D), then runs shorter than 2 s are absorbed
   by their neighbours. This turns noisy per-frame evidence into realistic possessions (the
   nearest-player baseline invents 669; this yields ~330 of mean ~16 s).
5. **Scope**: only the published half boundaries (DATA.md) are labelled; every other frame is `X`.
   No `X` is written inside the halves.

## Assets used / dropped
- Used: tactical ATD (`*_atd.txt`; `*_atd.xml` only for each track slot's team).
- **Smart Tagging — tried and dropped** (`solution/tags.py`, off by default, `--tags` to enable). Each
  phase has a Team; for `DEFENDING …`/`DEFENSIVE TRANSITION` it is the *defending* side, so possession
  is the opposite (the model agreed with those tags only 12–32 % un-inverted, 68–88 % inverted). I
  weighted each code by its label-free agreement with the model (two noisy annotators ⇒ implied tag
  accuracy ≈ 0.64–0.90). Checking that assumption against the public 5-minute window showed the tags
  agree with real possession only 65.5 % (several codes ≈ 50 %): the tags are not independent of the
  model and a phase is not a possession, so the weights were overconfident. Dropped.
- Not used: videos, homography, broadcast recording.
- A hand-built rule prototype (`solution/model.py`) was written first; it is not used.

## Training data
All labels come from **other matches**; none from the provided game.
- Metrica Sports open sample data, games 1, 2, 3 (https://github.com/metrica-sports/sample-data,
  anonymised, 25 fps tracking + human-coded event data; ~425k labelled frames). Credit: Metrica Sports.
- Possession/dead labels derived **algorithmically** from those events (`solution/external.py`):
  team of the latest PASS / CARRY / SHOT / RECOVERY / SET PIECE; dead ball from a BALL OUT, FAULT
  RECEIVED, goal, offside or hand-ball event to the next SET PIECE. They approximate, and are not
  identical to, the organisers' tracking-based definition.
- Training frames are perturbed to resemble the ATD. Ball jitter 0.8 m (player 0.3 m) was set by
  matching the distribution of the ball-to-nearest-player distance to the *unlabelled* ATD (a
  distribution match, no labels); ~7 % player drop-outs in 2 s blocks.

## Validation (provided game's labels never used for training or selection)
- Leave-one-game-out over the three games: team accuracy on live frames 86–88 %; dead-ball AUC
  0.90–0.92 overall but only 0.70–0.77 on frames where the ball is visible — the honest number,
  because the overall one is inflated by the ball-untracked-when-dead artifact in those games.
  Decoder settings were chosen on these held-out folds with the official `spellF1`; the optimum is a
  flat plateau (0.53 ± 0.003), so the choice is not sensitive.
- The public 5-minute window was used only to check alignment/orientation and to test one
  assumption (the tag reliability above), never to select parameters. For reference there:
  macroF1 0.708, spellF1 0.263 (floor baseline 0.621 / 0.326). One possession moves spellF1 by
  ~0.06 on 17 possessions, so this cannot rank anything (EVALUATION.md).
- Full match, no labels needed: 327 possessions of mean 15.8 s (public GT mean 14.2 s, not tuned to),
  H 77.1k / A 52.3k / D 29.4k graded frames.

## Possession rules implemented
Possession is the team the control model assigns the ball to, held by the decoder until sustained
evidence (the switch cost) says otherwise — loose balls, aerial duels and deflections therefore stay
with the previous team. Dead ball follows the dead model (slow collective movement, ball at/over the
edge, ball unseen for long, ball slow near a boundary). Restarts are not handled specially.

## Limitations
- **Dead ball is under-detected** (14.8 % of graded frames; the public window has 19.6 %). Stoppages
  where the ball stays visible (a player retrieving it) are hard: in the public window about two
  thirds of true dead frames are called live. Tried and rejected on held-out data: summing the two
  dead-ball experts (no AUC gain) and an EM prior-shift correction (the estimate, 3 %, was
  contradicted by the public window — the model is not calibrated enough under domain shift).
- Training labels come from events of two providers' sample games, approximating the organisers'
  tracking-based definition; the ATD is noisier than the training tracking.
- Ball-missing stretches rely on team shape and context only; no video is used.
- With more time: more open training games (SkillCorner, PFF), a sequence model instead of windowed
  features, restart detection from the ball track, a learned use of Smart Tagging.

## Reproduction
```
python3 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
# put the provided bundle in data/ (see DATA.md), then:
./make_submission.sh submission.csv     # downloads Metrica games 1-3, trains, predicts, validates
```
Deterministic: two from-scratch runs give a byte-identical CSV (checked by MD5). Python 3.15.0,
pinned versions in `requirements-solution.txt`. Training takes ~1 minute, prediction a few seconds.
