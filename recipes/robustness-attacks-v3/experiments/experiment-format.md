# Experiment File Format Specification

This format is parsed by the robustness dashboard (`robustness_dashboard/experiments.py`) to auto-load experiments into the UI. Follow it exactly when creating new experiment files.

## Required fields

Every experiment `.md` file in this directory MUST contain these fields using the exact format `**Field name:** value`. The dashboard parses these with regex — deviations will cause the experiment to not appear in the dropdown.

| Field | Format | Example |
|---|---|---|
| Title | `# ` H1 heading (first line) | `# Experiment 3: Hard token cap` |
| Status | `**Status:** <status>` | `**Status:** RUNNING` |
| Cluster | `**Cluster:** <name>` | `**Cluster:** oci` |
| Output path | `**Output path:** <full lustre path>` | `**Output path:** /lustre/fsw/.../gpqa-diamond-qwen8b-v3-exp3` |

## Optional fields (displayed in dashboard if present)

| Field | Format | Example |
|---|---|---|
| Hypothesis | `**Hypothesis:** <text>` | `**Hypothesis:** Truncating distractors eliminates false positives.` |
| Config | `**Config:** <text>` | `**Config:** Qwen3-8B, gpt-oss-120b, 8 GPUs, v3 pipeline, 100 iterations` |
| Progress | `**Progress:** <text>` | `**Progress:** iter 1-50 scored` |
| Results | `**Results:** <text>` | `**Results:** _pending_` |

## Rules

1. Use `**Field:** value` format (bold key, colon, space, value).
2. Each field should be on its own line, separated by blank lines for readability.
3. Output path must be the full `/lustre/...` path — not `/workspace/...`.
4. Cluster must be one of: `ord`, `oci`.
5. Status should start with one of: `RUNNING`, `PAUSED`, `COMPLETE`, `FAILED`, `SUBMITTED`.
6. File name should be `experiment_N.md` (or `baseline.md` for the reference run).
7. `TODO.md` and `experiment-format.md` are ignored by the parser.
8. Additional sections (Changes, Launch command, Issues, etc.) are free-form and not parsed — place them after the required fields.

## Field order

Follow this order for consistency across experiment files:

1. `# Title`
2. `**Hypothesis:**` (optional, omit for baseline)
3. `**Status:**`
4. `**Cluster:**`
5. `**Output path:**`
6. `**Changes:**` (free-form, not parsed)
7. `**Config:**`
8. `**Launch command:**` (free-form, not parsed)
9. `**Progress:**`
10. `**Results:**`
11. Additional sections (issues, analysis, etc.)

## Template

```markdown
# Experiment N: Short description

**Hypothesis:** One sentence stating what you expect to observe.

**Status:** SUBMITTED

**Cluster:** oci

**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-expN

**Changes:**
- Change 1
- Change 2

**Config:** Qwen3-8B eval target, mutation model, GPU count, pipeline version, iteration count

**Launch command:**
\```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-expN \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-expN \
    --mutation-model /hf_models/gpt-oss-120b \
    --benchmark gpqa \
    --subset diamond
\```

**Progress:** _not started_

**Results:** _pending_
```
