# Experiment File Format Specification

This format is parsed by the robustness dashboard (`robustness_dashboard/experiments.py`) to auto-load experiments into the UI. Follow it exactly when creating new experiment files.

## Required fields

Every experiment `.md` file in this directory MUST contain these fields, each on its own line using the exact format `**Field name:** value`. The dashboard parses these with regex — deviations will cause the experiment to not appear in the dropdown.

| Field | Format | Example |
|---|---|---|
| Title | `# ` H1 heading (first line) | `# Experiment 3: Longer distractors` |
| Cluster | `**Cluster:** <name>` | `**Cluster:** oci` |
| Output path | `**Output path:** <full lustre path>` | `**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp3` |
| Status | `**Status:** <status>` | `**Status:** RUNNING` |

## Optional fields (displayed in dashboard if present)

| Field | Format | Example |
|---|---|---|
| Hypothesis | `**Hypothesis:** <text>` | `**Hypothesis:** Longer distractors degrade accuracy more.` |
| Config | `**Config:** <text>` | `**Config:** Qwen3-8B, gpt-oss-120b, 8 GPUs, 100 iterations` |
| Progress | `**Progress:** <text>` | `**Progress:** iter 1-50 scored` |
| Results | `**Results:** <text>` | `**Results:** _pending_` |

## Rules

1. Use `**Field:**` format (bold key, colon, space, value) — NOT `- **Field:**` list format.
2. Output path must be the full `/lustre/...` path — not `/workspace/...`.
3. Cluster must be one of: `ord`, `oci`.
4. Status should start with one of: `RUNNING`, `PAUSED`, `COMPLETE`, `FAILED`, `SUBMITTED`.
5. File name should be `experiment_N.md` (or `baseline.md` for the reference run).
6. `TODO.md` and `experiment-format.md` are ignored by the parser.
7. Additional sections (Changes, Launch command, Issues, etc.) are free-form and not parsed.

## Template

```markdown
# Experiment N: Short description

**Hypothesis:** One sentence stating what you expect to observe.

**Changes:**
- Change 1
- Change 2

**Config:** Model, mutation model, GPU count, pipeline version, iteration count

**Status:** SUBMITTED
**Cluster:** oci
**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-expN

**Launch command:**
\```bash
python recipes/robustness-attacks-v3/launch_attack.py ...
\```

**Progress:** _not started_

**Results:** _pending_
```
