# Cross3B-DTEMA: tail-stabilized dual targets

## Evidence and hypothesis

Cross3B-DT2 increased the two-seed mean held-out return on Ant, Hopper, and
Walker2d by 5.0%, 23.6%, and 39.0%, but reduced AUC on Ant and Hopper. This
suggests a useful late policy with an early learning cost. Evaluation actor
EMA improved Hopper AUC and held-out return by 14.1% and 11.4%, but its fixed
lag reduced Ant AUC and Walker2d held-out return.

This pilot tests whether the two mechanisms are complementary. The proposed
variant preserves an exact CrossQ prefix through 100k critic updates, the end
of the existing BatchRenorm correction warmup. It then simultaneously:

1. averages two independent entropy-corrected next-action targets; and
2. evaluates a 25k-half-life EMA actor while retaining the online actor for
   collection and every training update.

The schedule depends only on update count. It does not use environment
identity, reward, evaluation return, or a tuned state threshold.

## Ablation and pilot

The fresh-seed pilot compares three conditions:

- CrossQ;
- DT2 from the start plus EMA after 100k (`dt2-ema`);
- both DT2 and EMA after 100k (`delayed-dt2-ema`).

The matrix is 3 conditions x 3 environments x seeds 18/19 = 18 runs. Each run
uses 300k steps, fixed five-episode evaluation every 25k steps, and an
independent 20-episode held-out evaluation. A candidate passes only if its mean
AUC and held-out return both exceed paired CrossQ on all three environments.

## Verification

```bash
cd /mnt/sda/rzq/cross3b
.venv/bin/python verify_dual_target_ema_agent.py
```

The check requires leaf-exact CrossQ actor, critic, temperature, and random
key during the prefix; a positive two-target dispersion after the boundary;
an altered critic; and a finite, distinct EMA evaluation action.

## Commands

```bash
cd /mnt/sda/rzq/cross3b
.venv/bin/python run_cross3b_dual_target_ema_pilot.py status \
  --group-dir runs/cross3b-dual-target-ema-pilot-20261102-r1
```

## Result

All 18 runs completed without divergence. Neither endpoint passed the strict
cross-environment rule, but the failures were complementary:

| Variant | Environment | AUC vs CrossQ | Held-out vs CrossQ | AUC wins | Held-out wins |
|---|---|---:|---:|---:|---:|
| DT2 + EMA | Ant | +5.3% | +20.2% | 1/2 | 1/2 |
| DT2 + EMA | Hopper | +26.4% | +10.3% | 2/2 | 1/2 |
| DT2 + EMA | Walker2d | -3.2% | -5.6% | 1/2 | 1/2 |
| Delayed DT2 + EMA | Ant | +5.6% | +18.2% | 1/2 | 2/2 |
| Delayed DT2 + EMA | Hopper | +16.6% | -10.5% | 1/2 | 1/2 |
| Delayed DT2 + EMA | Walker2d | +8.3% | +1.4% | 2/2 | 1/2 |

Ant improved at both onset endpoints. Hopper preferred immediate DT2, whereas
Walker2d preferred the 100k onset. This localizes the remaining sensitivity to
the target-sampling onset rather than the EMA implementation.

## Midpoint onset screen

The only follow-up tests the bracket midpoint: DT2 starts at 50k critic
updates and EMA remains fixed at 100k. It uses untouched seeds 20/21 and newly
run paired CrossQ controls. No environment identity or observed performance
enters the schedule.

```bash
.venv/bin/python run_cross3b_dual_target_ema_pilot.py status \
  --group-dir runs/cross3b-dual-target-ema-midstart-20261102-r1
```

The untouched seeds 20/21 screen was the first candidate to improve both mean
AUC and mean held-out return on all three environments:

| Environment | AUC vs CrossQ | Held-out vs CrossQ | AUC wins | Held-out wins |
|---|---:|---:|---:|---:|
| Ant | +6.3% | +8.9% | 1/2 | 1/2 |
| Hopper | +20.6% | +35.9% | 2/2 | 2/2 |
| Walker2d | +46.3% | +7.9% | 2/2 | 2/2 |

Ant seed 21 was effectively tied rather than materially worse: -0.39% AUC and
-0.07% held-out return. The candidate therefore advances to five seeds.

## Five-seed validation

The validation uses seeds 20--24. Completed seeds 20/21 are linked read-only
from the frozen midpoint screen; only fresh paired seeds 22/23/24 are trained.
This gives 5 CrossQ and 5 Cross3B-DTEMA-50K runs per environment, or 30 logical
runs in total.

```bash
.venv/bin/python run_cross3b_dual_target_ema_pilot.py status \
  --group-dir runs/cross3b-dual-target-ema-five-seed-validation-20261103-r1
```

Acceptance still requires positive mean AUC and held-out change on every
environment. The five-seed report must also show paired win counts and may not
drop a completed seed.

All 30 logical runs completed without divergence. Cross3B-DTEMA-50K passed
the fixed rule on every environment:

| Environment | CrossQ AUC | DTEMA-50K AUC | Change / wins | CrossQ held-out | DTEMA-50K held-out | Change / wins |
|---|---:|---:|---:|---:|---:|---:|
| Ant | 3296.2 +/- 738.8 | 3553.4 +/- 868.9 | +7.8% / 3/5 | 5043.4 +/- 881.5 | 5811.5 +/- 340.4 | +15.2% / 4/5 |
| Hopper | 2501.9 +/- 384.1 | 2829.0 +/- 370.6 | +13.1% / 5/5 | 2676.3 +/- 551.4 | 3111.8 +/- 527.2 | +16.3% / 4/5 |
| Walker2d | 3281.6 +/- 445.0 | 4390.2 +/- 790.6 | +33.8% / 5/5 | 5482.6 +/- 557.5 | 5852.9 +/- 399.6 | +6.8% / 5/5 |

This is the first frozen Cross3B candidate in the study to beat paired CrossQ
in both mean AUC and held-out return on Ant, Hopper, and Walker2d at five
seeds. The claim is limited to the 300k-step protocol and these three tasks.

## One-million-step main experiment

The algorithm and schedules are frozen from the 300k validation. A new main
experiment tests whether the improvement persists over a 1M-step horizon:

- methods: paired CrossQ and Cross3B-DTEMA-50K;
- environments: Ant-v4, Hopper-v4, and Walker2d-v4;
- untouched seeds: 25--29;
- training budget: 1,000,000 environment steps;
- periodic evaluation: fixed five episodes every 10,000 steps;
- final evaluation: 20 held-out episodes;
- acceptance: positive mean AUC and held-out change on all three environments.

This protocol is separate from the legacy 1M table, which evaluated only one
episode per 10k checkpoint. Newly run paired CrossQ controls prevent the two
evaluation protocols from being mixed.

```bash
.venv/bin/python run_cross3b_dual_target_ema_pilot.py status \
  --group-dir runs/cross3b-dual-target-ema-1m-main-20261103-r1
```

## Fixed simulation examples

The examples use the final EMA actor from candidate seed 20 and simulation
seed `1000020`. They were selected before rendering, not by rollout return.
All videos are H.264 MP4, 640 x 480, 50 fps, and 1000 frames.

| Environment | Return | Length | Final x | Video |
|---|---:|---:|---:|---|
| Ant | 6137.8 | 1000 | 279.95 | [MP4](runs/cross3b-dual-target-ema-five-seed-validation-20261103-r1/videos/cross3b-dtema50k_Ant-v4_seed20.mp4) |
| Hopper | 3389.5 | 1000 | 19.12 | [MP4](runs/cross3b-dual-target-ema-five-seed-validation-20261103-r1/videos/cross3b-dtema50k_Hopper-v4_seed20.mp4) |
| Walker2d | 6185.1 | 1000 | 41.50 | [MP4](runs/cross3b-dual-target-ema-five-seed-validation-20261103-r1/videos/cross3b-dtema50k_Walker2d-v4_seed20.mp4) |
