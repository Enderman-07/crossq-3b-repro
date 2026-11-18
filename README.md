# CrossQ-3B experiment materials

This private repository contains the CrossQ-3B experiment source bundle,
the matching 1M-step experiment notes, and the midterm report.

## Contents

- `crossq-3b-repro-source.tar.gz`: project source files, CrossQ third-party
  source and license, experiment runners, and focused verification scripts.
- `EXPERIMENT_CROSS3B_ORIGINAL_EMA_20261105.md`: protocol and matched
  six-method results for Ant-v4, Hopper-v4, and Walker2d-v4.
- `EXPERIMENT_CROSS3B_DUAL_TARGET_EMA_20261102.md`: dual-target EMA
  experiment notes.
- `CrossQ-3B_中期报告_20261109.html`: midterm report with the result tables
  and interpretation.

Extract the source bundle from the repository root with:

```bash
tar -xzf crossq-3b-repro-source.tar.gz
```

The report summarizes 90 matched runs (six methods, three environments, five
seeds, and 1M environment steps per run). The bundle does not include raw
per-run logs, checkpoints, virtual environments, or the unrelated aggregate
CSV inventory. Treat the report tables as summarized results; the underlying
run-level data is not included here for independent re-aggregation.

The official CrossQ source is included under `third_party/CrossQ` with its
license and attribution. Refer to that license before redistributing its
contents.
