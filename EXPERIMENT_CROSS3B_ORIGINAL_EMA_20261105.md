# Original actor-step Cross3B + EMA (2026-11-05)

## Purpose

This experiment returns to the exact structural hypothesis in the proposal:

- non-actor critic steps use the official CrossQ 2N update;
- every third critic step is also an actor step and uses one joint
  `B=(s,a) / T=(s',a') / P=(s,pi(s))` critic forward;
- critic and actor gradients on that step are computed from the same
  pre-update critic snapshot;
- the policy batch has full statistical mass, so this is a literal 3N update;
- actor EMA affects periodic evaluation, held-out evaluation, and deployment
  checkpoints only.

This is not the older `post_update_eval + actor_only` variant: that path uses
2N on non-actor steps but evaluates the actor against the updated critic.

## Frozen conditions

| Condition | Policy batch onset | EMA onset | EMA half-life |
|---|---:|---:|---:|
| `original3b-ema` | 0 critic updates | 100k | 25k |
| `staged-original3b-ema-50k` | 50k critic updates | 100k | 25k |

Both conditions otherwise use the same 1M protocol as the existing CrossQ and
CrossQ+EMA controls: Ant-v4, Hopper-v4, Walker2d-v4; seeds 25--29; fixed five-
episode evaluation every 10k steps; final 20 held-out episodes; batch 256;
2048-wide critic; policy delay 3; learning rate 0.001.

The primary comparison is each condition against matched CrossQ+EMA. A
condition passes only if mean AUC and held-out return improve on every
environment, with at least 3/5 paired wins per environment for both metrics.

## Verification

```bash
cd /mnt/sda/rzq/cross3b
.venv/bin/python verify_original_3b_actor_loss.py
.venv/bin/python verify_original_3b_ema_agent.py
```

The first verification checks that actor loss reads only the policy third of
the joint `3N` critic output and reduces across the critic axis. The agent
verification requires a leaf-exact CrossQ prefix before the policy batch
onset, a changed online update after onset, an identical online path under
different EMA schedules, and a finite distinct EMA evaluation action.

## Commands

```bash
cd /mnt/sda/rzq/cross3b
.venv/bin/python run_cross3b_original_ema_main.py status \
  --group-dir runs/cross3b-original-ema-1m-20261105-r2
```

To run the same original 3B update without evaluation EMA, prepare a
single-condition matrix with `--no-ema-only`. The runner places EMA onset one
step beyond the training budget, so periodic and held-out evaluation use the
online actor throughout:

```bash
.venv/bin/python run_cross3b_original_ema_main.py prepare \
  --group-dir runs/cross3b-original-no-ema-1m-20261106-r1 \
  --no-ema-only
.venv/bin/python run_cross3b_original_ema_main.py run \
  --group-dir runs/cross3b-original-no-ema-1m-20261106-r1 \
  --gpus 0 1 --workers-per-gpu 2
```

## Launch status

- Invalid group: `runs/cross3b-original-ema-1m-20261105-r1`. It was stopped
  after detecting that the joint actor loss indexed the critic axis as if it
  were the three-way batch axis. Its results must not be reported.
- Corrected group: `runs/cross3b-original-ema-1m-20261105-r2`.
- Matrix: 2 schedules x 3 environments x seeds 25--29 = 30 runs.
- Execution: GPUs 0/1, two workers per GPU.
- Launched: 2026-11-05 18:41 Asia/Shanghai in tmux session
  `cross3b_original_r2`.
- Before the corrected launch, actor-loss batch selection, gradient
  boundaries, CrossQ prefix, policy-batch scheduling, EMA isolation, and
  finite-action checks all passed.
- Initial audit: all four first-wave Walker2d runs reached 20k steps with
  finite metrics and two saved evaluations; no runtime or non-finite errors
  were detected.

## Original 3B without EMA

The matched no-EMA matrix completed all 15 runs in
`runs/cross3b-original-no-ema-1m-20261106-r1`: Ant-v4, Hopper-v4, and
Walker2d-v4; seeds 25--29; 1M steps; five fixed evaluation episodes every
10k steps; and 20 held-out episodes. EMA onset was set to 1,000,001 updates,
so EMA remained inactive for every training and evaluation run. All runs
completed without numerical divergence.

The following values are ratios of the five-seed method mean to the matched
CrossQ mean under the same evaluation protocol. Paired wins count seeds where
Original 3B without EMA exceeded CrossQ.

| Environment | Final (ratio / wins) | AUC (ratio / wins) | Held-out (ratio / wins) |
|---|---:|---:|---:|
| Ant-v4 | 107.7% / 5/5 | 118.0% / 5/5 | 105.2% / 5/5 |
| Hopper-v4 | 110.9% / 3/5 | 96.5% / 2/5 | 120.6% / 3/5 |
| Walker2d-v4 | 92.9% / 1/5 | 93.8% / 1/5 | 89.8% / 1/5 |

Original 3B without EMA improves all three aggregate metrics on Ant and
improves Hopper Final and held-out return, but reduces Hopper AUC. It regresses
on all three Walker2d means, with only one paired win out of five per metric.
The result therefore confirms environment dependence rather than a uniform
benefit from the original third-batch update.

## Matched 1M comparison across methods

The table below compares all six methods under the matched seeds 25--29,
1M-step, five-evaluation-episodes-per-10k protocol. Each cell is the ratio of
the method's five-seed mean to the matched CrossQ five-seed mean for that
environment and metric; CrossQ is 100%. `Best` is the mean
`best_checkpoint_return` recorded per run, not the held-out evaluation.

| Environment | Method | Final | AUC | Held-out | Best |
|---|---|---:|---:|---:|---:|
| Ant | CrossQ | 100.0% | 100.0% | 100.0% | 100.0% |
| Ant | Original 3B | 107.7% | 118.0% | 105.2% | 105.5% |
| Ant | CrossQ + EMA | 102.6% | 100.7% | 100.7% | 100.5% |
| Ant | Original 3B + EMA | 107.4% | 118.9% | 106.8% | 105.6% |
| Ant | 50k staged Original 3B + EMA | 103.9% | 109.4% | 102.4% | 101.3% |
| Ant | TDEMA-50K | 97.9% | 99.9% | 96.9% | 96.9% |
| Hopper | CrossQ | 100.0% | 100.0% | 100.0% | 100.0% |
| Hopper | Original 3B | 110.9% | 96.5% | 120.6% | 100.6% |
| Hopper | CrossQ + EMA | 126.2% | 120.3% | 137.9% | 99.8% |
| Hopper | Original 3B + EMA | 128.7% | 114.6% | 147.6% | 98.0% |
| Hopper | 50k staged Original 3B + EMA | 116.8% | 114.0% | 130.8% | 99.7% |
| Hopper | TDEMA-50K | 129.2% | 115.9% | 141.6% | 101.1% |
| Walker2d | CrossQ | 100.0% | 100.0% | 100.0% | 100.0% |
| Walker2d | Original 3B | 92.9% | 93.8% | 89.8% | 91.9% |
| Walker2d | CrossQ + EMA | 103.1% | 105.4% | 102.1% | 100.4% |
| Walker2d | Original 3B + EMA | 94.4% | 97.8% | 93.8% | 91.8% |
| Walker2d | 50k staged Original 3B + EMA | 102.0% | 106.8% | 100.9% | 99.5% |
| Walker2d | TDEMA-50K | 101.8% | 106.1% | 100.6% | 98.7% |

These are ratios of means, not averages of per-seed ratios. The corresponding
run summaries are under:

- CrossQ and TDEMA-50K: `./runs/cross3b-dual-target-ema-1m-main-20261103-r1/`
- CrossQ + EMA: `./runs/crossq-ema-1m-control-20261103-r1/`
- Original 3B without EMA: `./runs/cross3b-original-no-ema-1m-20261106-r1/`
- Original 3B + EMA and staged Original 3B + EMA:
  `./runs/cross3b-original-ema-1m-20261105-r2/`
