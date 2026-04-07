# Robustness Attacks v3 — Experiments

## Baseline
- **Run:** `gpqa-diamond-qwen8b-new-mutation-prompt` (v2, ord cluster, 100 iterations)
- **Config:** Qwen3-8B, gpt-oss-120b mutation model, 8 GPUs eval, original few-shot examples
- **Path:** `/lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-new-mutation-prompt`

> Note: Baseline was run with v2 pipeline. Not directly re-runnable with v3.

---

## Experiment 1: Improved few-shot examples (50-60 token target)

**Hypothesis:** Current few-shot examples have wildly inconsistent token lengths (16–514 tokens, only 16% in 50-60 range). Rewriting all 100 examples to consistently hit 50-60 tokens will produce more uniform distractor lengths and potentially stronger attacks.

**Changes:**
- Rewrote all 100 few-shot examples across 5 files (code_snippet, encrypted_text, markup_noise, math_fact, random_fact)
- Each `mutated_distractor` verified to be 50-60 Qwen3-8B tokens
- Eliminated duplicate content (e.g., repeated octopus facts)
- Ensured quality: each mutation matches target type, is coherent, and is a plausible distractor

**Config:** Same as baseline (Qwen3-8B, gpt-oss-120b, 8 GPUs, v3 pipeline), 100 iterations on ord

**Status:** SUBMITTED
**Cluster:** ord
**Expname prefix:** `robustness-v3-exp1`
**Output path:** `/workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp1`

**Launch command:**
```bash
python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster ord \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp1 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --expname-prefix robustness-v3-exp1 \
    --mutation-model /hf_models/gpt-oss-120b \
    --benchmark gpqa \
    --subset diamond
```

**Results:** _pending_

---

## Experiment 2: Claude Sonnet 4.5 as mutator (via NVIDIA inference API)

**Hypothesis:** Using a stronger, instruction-following model (Claude Sonnet 4.5) as the mutator — instead of gpt-oss-120b — may produce higher-quality distractor mutations that better follow the mutation prompt's constraints (type transformation, conciseness, adversarial plausibility). This could lead to faster accuracy degradation per iteration.

**Changes:**
- Added `--mutation-api-url`, `--mutation-api-model`, `--mutation-api-key-env` args to `step1.py` and `launch_attack.py`
- Added `--few-shot-dir` arg to select which few-shot examples to use
- Uses NVIDIA inference API (`https://inference-api.nvidia.com`) with model `azure/anthropic/claude-sonnet-4-5`
- Uses **original** few-shot examples (`prompts/few-shot-examples-original/`, copied from v2) to isolate mutator effect
- Mutation runs on CPU partition (no GPU needed for step1)
- `seed` parameter disabled for external APIs (not supported by all providers)

**Config:** Qwen3-8B eval target, Claude Sonnet 4.5 mutator via NVIDIA API, original few-shot examples, 8 GPUs eval, v3 pipeline, 50 iterations on ord

**Status:** SUBMITTED
**Cluster:** ord
**Expname prefix:** `robustness-v3-exp2`
**Output path:** `/workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp2`

**Launch command:**
```bash
python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster ord \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp2 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 50 \
    --expname-prefix robustness-v3-exp2 \
    --mutation-api-url https://inference-api.nvidia.com \
    --mutation-api-model azure/anthropic/claude-sonnet-4-5 \
    --few-shot-dir /nemo_run/code/recipes/robustness-attacks-v3/prompts/few-shot-examples-original \
    --benchmark gpqa \
    --subset diamond
```

**Results:** _pending_
