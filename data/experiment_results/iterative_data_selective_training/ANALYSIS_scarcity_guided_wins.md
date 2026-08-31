# Scarcity makes guidance pay across the loop — analysis (2026-08-19)

Session goal: test the paper storyline's final act — do the four mix policies
(Constant, Exponential dump ×0.1, Adaptive severity, Adaptive size) turn
"recovery to parity" into an actual across-the-loop guided win when the model is
less pretrained and/or the candidate pool is limited? All sweeps: new-only
regime, warm-start, Quantile Regression detector,
configs in `scripts/iterative/sweep_configs/`.

## Storyline verification against existing data (pre-session sweeps)

1. **Round-1 win, then decay — confirmed.** Conditions map: iteration-1 gap ≈ 0
   to −0.086, iterations 2+ lose everywhere; IoU 0.21 → 0.05–0.08.
2. **Aggressive mix drop-off — confirmed.** Exponential ×0.1 from α₀=1 recovers
   exact parity (auc −0.001).
3. **Wording correction for the paper:** adaptive-on-**size** (region area)
   FAILED (+0.024/+0.079); adaptive-on-**severity** (in/out error contrast =
   "how pronounced") reached parity (−0.002 at α₀=0.5). The storyline's "less
   pronounced → closer to random" is the severity gate.
4. **Undertraining alone does NOT rescue constant mix** (iter20_anchor_i3n50:
   auc +0.029 vs +0.021 at the default anchor). The rescue has to come from
   schedule × scarcity.

## New sweep 1 — `sweep__iter_guided_wins_search.csv` (960 traj, 5 seeds)

Grid: 4 schedules × α₀{0.5,1} × iters_initial{3,12} × n_train{30,100} ×
n_select{30,100} × pool_deficit{0, 0.9, 0.98}, K=8.

- **pool_deficit=0.98 is the discovery**: only ~8 of 2000 candidates per round
  fall inside the gap (radius 0.25), so random picks ~0.4 gap points/round while
  guided harvests them deliberately → a persistent advantage.
- Top cell (Constant α=0.5, i12, n_train 30, n_select 100, deficit 0.98):
  auc −0.043, gap compounding −0.013→−0.086 over 8 rounds, 5/5 seeds, IoU held
  at 0.12–0.20. (Partly seed luck, see confirmation.)
- iters_initial (3 vs 12) barely matters; α₀=1.0 loses everywhere; the levers
  are pool_deficit, n_train small, n_select large.

## New sweep 2 — `sweep__iter_scarcity_confirm.csv` (15 seeds, K=8)

4 schedules × deficit {0, 0.9, 0.95, 0.98, 0.995} at α=0.5, n_train 30,
n_select 100.

- **Sign flip is real but modest:** at deficit 0.98 Constant auc −0.018±0.012
  (p=0.16, 10/15 seeds ahead) and Exponential −0.019±0.014; at deficits
  0–0.95 wash or loss. Adaptive severity ~ −0.005 (never harmful, never best).
- **Mechanism confirms:** detected severity GROWS over the loop at high deficit
  (1.9→3.2 at 0.98; →5.3 at 0.995) instead of collapsing; IoU stays 0.13–0.19.
  A persistent real target exists — constant focus is finally warranted.
- **Interior optimum in deficit:** 0.995 is too scarce (even guided can't
  harvest → both arms keep the weakspot, gap shrinks). 0.98 is the sweet spot.
- **Geometric bound bites:** whole-area gap ≈ area_frac × in-weakspot cut;
  at radius 0.25 (~15% area) a −0.10 err_in lead buys only ~−0.02 MAE.

## New sweep 3 — `sweep__iter_scarcity_longrun.csv` (K=20, deficit 0.98, 15 seeds)

Constant & Adaptive severity at the confirmed cell.

- Guided lead does NOT compound forever: it is front-loaded (rounds 1–2:
  −0.032/−0.026) then hovers near parity with guided slightly ahead
  (auc −0.007, 10/15 seeds; final-iter −0.005).
- **In-weakspot advantage persists the whole loop:** err_in 0.66 vs 0.77 at
  K=20 (gap −0.10 every round); whole-area MAE 0.243 vs 0.248.

## New sweep 4 — `sweep__iter_showcase_search.csv` (RUNNING at checkpoint time)

Multiplies the geometric bound instead of fighting it: radius {0.25, 0.35}
(0.35 ≈ 38% of area) × sel_sigma {0.15, 0.3, 0.5} × α {0.5, 0.75} ×
deficit {0.9, 0.98} × {Constant, Adaptive severity}, n_train 30, n_select 100,
K=8, 15 seeds = 720 trajectories. All session scripts are saved in
`analysis_scripts/` next to this file (headless runner `run_iter_sweep.py`,
per-sweep analyses, `make_scarcity_figure.py` →
`figures/Exploration/fig_scarcity_guided_wins.png`). If interrupted, re-run
`python analysis_scripts/run_iter_sweep.py iter_showcase_search --workers 14`
(resumable) then `analyze_showcase.py`.

## New sweep 4 results — `sweep__iter_showcase_search.csv` (15 seeds, K=8)

radius {0.25, 0.35} × σ {0.15, 0.3, 0.5} × α {0.5, 0.75} × deficit {0.9, 0.98}
× {Constant, Adaptive severity}. First statistically significant across-loop
guided wins: **Adaptive severity, α₀=0.5, σ=0.15, radius 0.25, deficit 0.98 →
auc −0.035±0.014 (p=0.028)**, final-iter −0.049 (p=0.021), 80% seeds; and
radius 0.35/σ=0.30 → −0.024±0.007 (**p=0.005**, 13/15 seeds). Crucial contrast:
a narrow kernel with CONSTANT mix fails (+0.001) — the severity gate is what
makes hard focus safe. α₀=0.75 worse than 0.5; deficit 0.98 ≫ 0.9; radius 0.35
did not multiply the whole-area effect as the geometric bound predicted (the
detected region, not the induced one, sets area_frac).

## Money run — `sweep__iter_money_policies.csv` (30 seeds, K=8) and
## `sweep__iter_money_longrun.csv` (15 seeds, K=20)

Four policies × {radius 0.25/0.35} × {σ 0.15/0.30} at deficit 0.98, n_train 30,
n_select 100, α₀ 0.5. Figure: `figures/Exploration/fig_money_policies.png`.

At the tight geometry (radius 0.25, σ 0.15), across-loop gap (30 seeds):

| policy              | auc gap | p | seeds ahead |
|---------------------|---------|-----|----|
| **Exponential dump ×0.1** | **−0.058±0.009** | <1e-5 | **93%** |
| Adaptive (severity) | −0.030±0.010 | 0.004 | 67% |
| Constant            | +0.031±0.011 | 0.007 (worse) | 30% |
| Adaptive (size)     | +0.063±0.011 | <1e-4 (worse) | 3% |

Mechanism of the dump's win: under deficit 0.98 the random arm can never fix
the gap later, so the big round-1 repair (narrow kernel, full α₀ share) is
**locked in** — front-load, then coast. But the dump is geometry-fragile
(+0.003 at radius 0.35/σ 0.30), while **Adaptive severity is guided-ahead in
all 8 geometry×horizon cells** and is the only policy still ahead at K=20
(auc −0.022, p=0.047 at r0.25/σ0.15 and p=0.019 at r0.35/σ0.30; in-weakspot
gap negative in every one of 20 rounds; realised α hovers 0.2–0.3). Constant
collapses at K=20 (+0.05…+0.13, 0% seeds ahead).

## Protocol trade-off (round-1 visibility probe, 2026-08-20)

Attempting to make the round-1 whole-area win visible in the paper's Setting-1
figure, we crossed kernel width {0.15, 0.3, 0.5} × retrain budget {100, 400} ×
early stopping {on, off} (sweeps `iter_final_*`, `iter_final3_*`,
`iter_round1_probe`, `iter_es_*`). **Finding: no protocol satisfies both ends
of the paper.** A whole-area round-1 win needs the retraining to stay
unconverged — early stopping (the companion's protocol) gives it1 −0.128 — but
the scarcity victory needs guided's concentrated batch fully absorbed, and
under early stopping the scarce setting collapses entirely (everything loses;
guided's concentrated 10-point validation split cripples its retraining).
σ=0.15 without ES strengthens the loss and the scarcity win (and works at
n_train 100, one-knob settings) but puts round 1 slightly behind on the whole
area (+0.032) and the adaptive gate only halves the Setting-1 loss (+0.046).
The always-available honest round-1 signal is **in-weakspot**, in every
protocol. Decision (user, 2026-08-20): keep σ=0.5 / retrain-400 / no-ES for
Settings 1–2 and the money-run cell for Setting 3; the paper text points the
reader to the in-weakspot panel for round 1.

## Paper-ready storyline status

Single round works → repetition surprisingly hurts → mechanism: gap filled in
round 1, detector re-fires on noise (IoU collapse) → fix A: front-load α
(exp ×0.1) or severity-adaptive gate → recovery to parity → **final act: when
the data pool is the reason the weakspot exists (pool_deficit ≈ 0.98 — the
fixed-dataset industrial premise), the deficit persists, detection stays real,
and guided selection beats random across the loop with high significance.**
Undertraining is NOT the lever (iters_initial 3 vs 12 indistinguishable);
scarcity is, plus a small initial set (n_train 30).

**Recommended showcase operating point** (30-seed verified): pool_deficit 0.98,
n_train 30, n_select 100, α₀ 0.5, sel_sigma 0.15, radius 0.25, K=8, warm-start,
new-only, Quantile Regression. Headline: exponential dump ×0.1 → auc −0.058
(p<1e-5, 28/30 seeds); adaptive severity → −0.030 (p=0.004) and uniquely
persistent to K=20. Closing message: the right iterative policy is
*severity-gated focus* — it front-loads by itself when the deficit is
transient, keeps focus alive when scarcity makes the deficit persistent, and
never loses in either world; the size gate and constant focus both fail.
