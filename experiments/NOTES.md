# Experiment notes (Track C)

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
