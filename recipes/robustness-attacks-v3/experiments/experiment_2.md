# Experiment 2: Claude Sonnet 4.5 as mutator (via NVIDIA inference API)

**Hypothesis:** Using a stronger, instruction-following model (Claude Sonnet 4.5) as the mutator — instead of gpt-oss-120b — may produce higher-quality distractor mutations that better follow the mutation prompt's constraints (type transformation, conciseness, adversarial plausibility). This could lead to faster accuracy degradation per iteration.

**Status:** PAUSED — iter 1-8 complete, needs relaunch to continue from iter 9

**Cluster:** oci

**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp2

**Changes:**
- Added `--mutation-api-url`, `--mutation-api-model`, `--mutation-api-key-env` args to `step1.py` and `launch_attack.py`
- Added `--few-shot-dir` arg to select which few-shot examples to use
- Uses NVIDIA inference API (`https://inference-api.nvidia.com`) with model `azure/anthropic/claude-sonnet-4-5`
- Uses **original** few-shot examples (`prompts/few-shot-examples-original/`, copied from v2) to isolate mutator effect
- Mutation runs on CPU partition (no GPU needed for step1)
- `seed` parameter disabled for external APIs (not supported by all providers)
- System prompt added to reduce refusals (~10 fallbacks across 8 iterations, all involving ENCRYPTED_TEXT)

**Config:** Qwen3-8B eval target, Claude Sonnet 4.5 mutator via NVIDIA API, 8 GPUs eval, v3 pipeline, 50 iterations

**Few-shot examples:** original (v2, 16-514 tokens, inconsistent lengths)

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp2 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 50 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp2 \
    --mutation-api-url https://inference-api.nvidia.com \
    --mutation-api-model azure/anthropic/claude-sonnet-4-5 \
    --few-shot-dir /nemo_run/code/recipes/robustness-attacks-v3/prompts/few-shot-examples-original \
    --benchmark gpqa \
    --subset diamond
```

**Progress:** iter 1-8 scored, ~10 mutation fallbacks due to Claude refusals (launcher crashed due to OCI login node dc-02 going down)

**Results:** _pending_

---

## Issue: Distractor length explosion causing 88% no_answer rate

**Observed in iter 6, CODE_SNIPPET pos-0:**
- `no_answer: 88.4%`, `symbolic_correct: 1.0%`
- `avg_tokens: 29,405` (near 32K limit)
- `finish_reason: length` — model exhausts token budget on reasoning, never produces answer
- `_generation_finished_thinking: False` — `</think>` tag never reached

**Root cause:** Claude Sonnet 4.5 generates massive distractors that the original few-shot examples don't constrain:

| Slot | Distractor length |
|------|------------------|
| CODE_SNIPPET pos=0 | 850 chars |
| CODE_SNIPPET pos=1 | 953 chars |
| CODE_SNIPPET pos=2 | 1,251 chars |
| MARKUP_NOISE pos=2 | 2,111 chars |
| RANDOM_FACT pos=0 | 692 chars |
| ENCRYPTED_TEXT pos=0 | 758 chars |

Baseline distractors were ~50-100 chars. Claude produces 5-20x longer distractors because the original few-shot examples range from 8 to 514 tokens with no consistent length signal. When these huge distractors are prepended to GPQA questions, the total prompt is so long that Qwen3-8B spends all 32K generation tokens on chain-of-thought reasoning without ever reaching an answer.

**Options to fix:**
1. Stricter length constraint in mutation prompt
2. Use the new improved few-shot examples (50-60 tokens) for exp2 instead of the originals
3. Post-process truncation in step2 (cap at 200 chars)
