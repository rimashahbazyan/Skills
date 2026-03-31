# Robustness-Attacks — Output Folder Structure

All paths below are relative to `--output-folder` (referred to as `{OUT}`).
For the OCI reference run this is `/workspace/robustness-attacks/gpqa-diamond-qwen8b-10iter`.

---

## Top-level overview

```
{OUT}/
├── current_state.jsonl          ← active distractor grid (overwritten each iteration)
├── {benchmark}_{subset}.jsonl   ← local copy of benchmark rows (written once by step2)
│
├── archive/
│   └── iter{N}.jsonl            ← per-iteration snapshot (written by step1, enriched by step4)
│
├── iter{N}/                     ← per-iteration benchmark dataset dir (one per iteration N=1…K)
│   ├── __init__.py              ← benchmark config stub (written by step1 or _upload_iter_stubs)
│   └── {benchmark}_{subset}__type-{TYPE}__pos-{P}.jsonl   ← injected eval files (written by step2)
│
└── logs/                        ← ALL NeMo-Run job logs and eval output
    └── iter{N}/
        ├── step1/               ← NeMo-Run logs for step1 (mutation)
        ├── step2/               ← NeMo-Run logs for step2 (injection)
        ├── step3/               ← eval (step3) output, one sub-dir per (type, position)
        │   └── type-{TYPE}_pos-{P}/
        │       ├── iter{N}/
        │       │   └── output.jsonl         ← raw model predictions
        │       ├── eval-results/
        │       │   └── iter{N}/
        │       │       └── metrics.json     ← accuracy scores (read by step4)
        │       └── logs/                    ← NeMo-Run logs for this eval job
        └── step4/               ← NeMo-Run logs for step4 (selection)
```

---

## File-by-file reference

### `current_state.jsonl`
- **Written by**: step1 (initial creation, first iteration only), step4 (every iteration)
- **Read by**: step1 (next iteration, to get `parent_score` for lineage)
- **Contents**: 15 lines — one per grid slot `(type, position)` — with fields
  `id`, `distractor`, `type`, `position`, `parent_id`, `parent_score`, `eval_score` (after iter 1)

### `{benchmark}_{subset}.jsonl`  *(e.g. `gpqa_diamond.jsonl`)*
- **Written by**: step2 on the first iteration (copied from `/nemo_run/code/nemo_skills/dataset/`)
- **Read by**: step2 every iteration (reused as the canonical benchmark source)

---

### `archive/iter{N}.jsonl`
- **Written by**: step1 (immediately after mutation — no `eval_score` yet)
- **Enriched by**: step4 (adds `eval_score` field in-place after reading metrics)
- **Contents**: same schema as `current_state.jsonl`; provides a full history of every
  distractor state and its measured accuracy

---

### `iter{N}/__init__.py`
- **Written by**: step1 (inside the container) **or** `_upload_iter_stubs` (on the host,
  for local executor) — whichever runs first
- **Contents**: three-line NeMo benchmark config:
  ```python
  METRICS_TYPE = "multichoice"
  EVAL_SPLIT = "test"
  GENERATION_ARGS = "++eval_type=multichoice"
  ```
- **Read by**: `ns eval` to resolve the dynamic benchmark module

### `iter{N}/{benchmark}_{subset}__type-{TYPE}__pos-{P}.jsonl`
- **Count**: 5 types × 3 positions = **15 files per iteration**
- **Written by**: step2 (overwrites the empty stub created by `_upload_iter_stubs`)
- **Read by**: `ns eval` (step3) — each file becomes one eval job
- **Contents**: all benchmark rows with two extra fields injected:
  `distractor_id`, `distractor`

### `logs/iter{N}/step3/type-{TYPE}_pos-{P}/iter{N}/output.jsonl`
- **Written by**: `ns eval` — raw per-question model outputs
- One file per `(type, position)` pair → **15 output files per iteration**

### `logs/iter{N}/step3/type-{TYPE}_pos-{P}/eval-results/iter{N}/metrics.json`
- **Written by**: `ns eval` scorer
- **Read by**: step4 (`load_score`)
- **Key field**: `metrics["iter{N}"]["pass@1"]["symbolic_correct"]` → the accuracy
  score step4 uses for accept/revert decisions

### `logs/iter{N}/step3/type-{TYPE}_pos-{P}/logs/`
- **Written by**: NeMo-Run for the eval job of that `(type, position)` pair

---

### `logs/iter{N}/step{1,2,4}/`
- **Written by**: NeMo-Run for the CPU/GPU `run_cmd` jobs
- One directory per step per iteration, contains Slurm stdout/stderr

---

## NeMo-Run internal tracking  *(outside `{OUT}`)*

NeMo-Run writes experiment metadata to `job_dir` from the cluster config:

```
/lustre/fsw/portfolios/llmservice/users/rshahbazyan/nemo_logs/
└── {expname}/          ← one dir per submitted experiment
    ├── *.log
    └── nemo_run/
```

---

## Iteration data-flow summary

```
step1 ──writes──► archive/iter{N}.jsonl          (mutations, no scores)
               ► iter{N}/__init__.py

step2 ──writes──► iter{N}/{benchmark}_{subset}__type-*__pos-*.jsonl   (15 files)

step3 ──writes──► logs/iter{N}/step3/type-*_pos-*/iter{N}/output.jsonl
               ► logs/iter{N}/step3/type-*_pos-*/eval-results/iter{N}/metrics.json

step4 ──reads───  archive/iter{N}.jsonl  +  logs/iter{N}/step3/.../metrics.json  +  current_state.jsonl
      ──writes──► archive/iter{N}.jsonl          (adds eval_score in-place)
               ► current_state.jsonl             (selected / reverted grid for next iteration)
```

---

## Full expanded example  *(N=2, gpqa/diamond, 5 types × 3 positions)*

```
{OUT}/
├── current_state.jsonl
├── gpqa_diamond.jsonl
├── archive/
│   ├── iter1.jsonl
│   └── iter2.jsonl
├── iter1/
│   ├── __init__.py
│   ├── gpqa_diamond__type-CODE_SNIPPET__pos-0.jsonl
│   ├── gpqa_diamond__type-CODE_SNIPPET__pos-1.jsonl
│   ├── gpqa_diamond__type-CODE_SNIPPET__pos-2.jsonl
│   ├── gpqa_diamond__type-ENCRYPTED_TEXT__pos-0.jsonl
│   ├── gpqa_diamond__type-ENCRYPTED_TEXT__pos-1.jsonl
│   ├── gpqa_diamond__type-ENCRYPTED_TEXT__pos-2.jsonl
│   ├── gpqa_diamond__type-MARKUP_NOISE__pos-0.jsonl
│   ├── gpqa_diamond__type-MARKUP_NOISE__pos-1.jsonl
│   ├── gpqa_diamond__type-MARKUP_NOISE__pos-2.jsonl
│   ├── gpqa_diamond__type-MATH_FACT__pos-0.jsonl
│   ├── gpqa_diamond__type-MATH_FACT__pos-1.jsonl
│   ├── gpqa_diamond__type-MATH_FACT__pos-2.jsonl
│   ├── gpqa_diamond__type-RANDOM_FACT__pos-0.jsonl
│   ├── gpqa_diamond__type-RANDOM_FACT__pos-1.jsonl
│   └── gpqa_diamond__type-RANDOM_FACT__pos-2.jsonl
├── iter2/
│   └── ...   (same structure)
└── logs/
    ├── iter1/
    │   ├── step1/
    │   ├── step2/
    │   ├── step3/
    │   │   ├── type-CODE_SNIPPET_pos-0/
    │   │   │   ├── iter1/
    │   │   │   │   └── output.jsonl
    │   │   │   ├── eval-results/
    │   │   │   │   └── iter1/
    │   │   │   │       └── metrics.json
    │   │   │   └── logs/
    │   │   └── ...   (× 15 type-pos combos)
    │   └── step4/
    └── iter2/
        └── ...
```
