# Conditions map & mix adaptation — analysis (2026-08-19)

Three new sweeps (`sweep__iter_conditions_map.csv`, `sweep__iter_mix_adaptation.csv`,
`sweep__iter_mix_rate.csv`; configs in `scripts/iterative/sweep_configs/`), all
new-only regime, warm-start, K=8, Quantile Regression detector. Readable in the
**Iterative — Visualise Results** page. Summary figure:
`figures/fig_mix_adaptation_summary.png`.

## Question 1 — under which conditions does targeted (guided, warm-start) beat random?

Grid: mix α {0.25, 0.5, 1.0} × kernel σ {0.15, 0.3, 0.5} × gap radius
{0.12, 0.25, 0.35} × pool_deficit {0, 0.9}, 5 seeds (270 trajectories).

- **No condition wins across the loop.** Best cells are statistical ties
  (α=0.25 with σ=0.5, and the scarcity cells at α=0.25); everything else loses,
  monotonically worse with focus. Worst: α=1.0, σ=0.15 → +1.0…+1.5 MAE
  (collapse onto a noise target + catastrophic forgetting).
- **The guided advantage is a round-1 phenomenon.** Iteration-1 gap ≈ 0 to
  −0.086 (best: radius 0.35, deficit 0.9, σ=0.15, α=0.5); iterations 2+ are
  +0.18…+0.26 in the mean, in every condition block.
- **Mechanism confirmed (the "weakspot too small after round 1" hypothesis):**
  detection IoU with the true gap is 0.21 at round 1 and 0.05–0.09 from round 2
  on, with the detected centre ~0.3 away from the true gap — the gap is filled
  by round 1's selection and later rounds aim at noise.
- **Candidate scarcity (pool_deficit=0.9) softens but does not rescue:** it
  keeps detection slightly real for rounds 2–4 (IoU 0.09 vs 0.06), gives the
  strongest round-1 wins, and produces the only near-tie cells — but rounds 2+
  still lose.

## Question 2 — can adjusting α over the loop fix the multi-iteration failure?

α schedules × starting mix {0.5, 1.0} × decay rate, 10 seeds. Across-loop gap
(guided − random; negative = guided ahead), t vs 0:

| schedule (α₀=1.0)                        | auc gap  | verdict |
|------------------------------------------|----------|---------|
| Constant                                 | +0.081 (t=10.0) | fails badly |
| Linear decay → 0 by K                    | +0.026 (t=3.6)  | too slow, still loses |
| Exponential ×0.5/round                   | +0.003 (t=0.3)  | tie |
| **Exponential ×0.1/round** (≈ guided round 1 only) | **−0.001** (t=−0.1), final gap −0.009 | **recovers parity, keeps round-1 repair** |
| Adaptive (severity)                      | +0.021 (t=3.6)  | partial — α re-fires on noise spikes |
| Adaptive (size)                          | +0.079 (t=9.8)  | fails — size gate doesn't discriminate |

At α₀=0.5 the same ordering holds; **Adaptive (severity)** is the best overall
cell there (−0.002, final −0.008, 7/10 seeds guided ahead — not significant).

**Verdict.** The hypothesis is right and the fix works — but the honest claim is
*recovery, not victory*: front-loading α (exponential dump after round 1, or the
severity gate at moderate α₀) removes the entire multi-iteration penalty while
keeping the round-1 repair, ending at parity with random. Nothing sustains a
significant lead, because after round 1 there is no under-served region left to
exploit at this operating point; linear decay is always too slow. The paper-ready
framing: guidance pays exactly while a real deficit exists, and the correct
iterative policy is "focus hard, then hand the budget back to coverage as soon
as the detected severity collapses."
