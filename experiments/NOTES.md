# Experiment notes (Track C)

> **NEW YARDSTICK (read this first).** `cv_fast.py` now scores against a *controlled-possession* target by default:
> DFL labels with H/A runs shorter than 3 s merged into their live neighbour (`dfl.apply_hold`); the raw-DFL score is
> printed second. Numbers in `experiments/log.csv` carry a `[train hold | eval hold]` tag: rows tagged `eval hold 0.0s`
> or without a tag use the old raw target and are NOT comparable with `eval hold 3.0s` rows.
> Current main: decoder 0.4 / 6 / 25 / 50, trained on raw labels -> **spellF1 0.641 (3 s target), 0.442 (raw)**,
> macroF1 0.865, team accuracy 0.861.

Yardstick: `python solution/cv_fast.py --note "..."` (leave-one-match-out on the 7 DFL matches; ~30 s with a warm
feature cache, ~70 s cold). Each run appends a row to `experiments/log.csv`. Gate for accepting a change: mean spellF1
+0.01, macroF1 not lower, at most two folds worse.

## Baseline (main, 97 features, depth-5 / 250-iteration trees)
spellF1 0.468, macroF1 0.858, team accuracy 0.857, dead-ball AUC 0.989, ~262 predicted vs 437 true possessions.

## Decoder findings
* Parameter grid (144 cells): best 0.470 vs 0.468 -> plain Viterbi parameters are at their ceiling.
* Hidden semi-Markov decoder with explicit state durations fitted on the other matches' labels
  (`decode.decode_span_hsmm`): best 0.467 -> no gain, fails the gate. Kept in `decode.py` for the record, unused.
  Conclusion: the decoder is not the bottleneck; evidence quality is.

## Where the error is (out-of-fold, all 7 matches)
True-possession recall at IoU 0.5 by true duration:

| duration | share of possessions | matched |
|---|---|---|
| < 1 s | 13% | 1% |
| 1-3 s | 31% | 14% |
| 3-6 s | 23% | 36% |
| 6-15 s | 22% | 63% |
| > 15 s | 12% | 88% |

* 44% of DFL possessions are shorter than 3 s (touch-level possession flips); almost none are recovered. DFL's
  possession definition is finer than a "controlled possession" reading, so spellF1 on DFL has a low ceiling that is
  partly a label-granularity artefact. (The organisers' public window has fewer very short possessions: 5 of 17 under 3.5 s.)
* Frame confusion: H<->A about 13% of live frames; dead ball 94% correct. Team accuracy among frames both alive: 0.865.

## Implications for the team
* Track A/B: the team-control model (H vs A confusion) is the main fixable error. Ball-missing stretches are where it is weakest.
* Be careful not to over-optimise DFL spellF1 by chasing sub-3 s flips: the organisers' ground truth may not contain them.
  Prefer changes that raise macroF1 / team accuracy and long-possession recall.

## Hold study (label granularity) - 16 Nov
Why: DFL flips possession on any touch (437 possessions/match, 44% under 3 s); the challenge defines *controlled*
possession ("a player keeps the ball for a few seconds"; touches / deflections / aerial duels do not count).
* spellF1 by target hold with the OLD decoder (0.8/3/12/25), model trained on raw labels: raw 0.468, 2 s 0.575, 3 s 0.592, 4 s 0.581.
* Re-tuning the decoder for coarse targets is worth far more than relabelling: 0.4/6/25/50 gives 0.593 / 0.641 / 0.645 on
  2 / 3 / 4 s and 0.442 on raw (-0.026). Neighbouring settings are within 0.01.
* Training on hold-merged labels does not help once the decoder is re-tuned: mean over the 2/3/4 s targets =
  0.626 (raw labels) vs 0.616 (2 s) vs 0.615 (3 s). Keep training on raw labels.
* Caveat: on the public window the new file predicts 10 of 17 possessions (the real labels keep a few < 1.5 s). The
  2-4 s target range is an assumption from the challenge text, not a measurement; do not tune it on the public window.
* Keeper-slot swap CHECKED (label-free). Keeper slots are 23 (home) and 45 (away). Slot 45 sits >35 m from its own goal
  for 6,797 frames in half 1 and 9,594 in half 2 (slot 23: 348 / 1,249), in runs of ~1,300-5,700 frames, at the OTHER
  keeper's end: a phantom away player in the home box, not a pairwise swap. Blanking those frames (NaN) changes only
  813 of 158,857 graded labels (0.5%, mostly A/H -> D). It cannot be validated (DFL has no such artefact, public window
  too small), so it is NOT applied; submission unchanged.
