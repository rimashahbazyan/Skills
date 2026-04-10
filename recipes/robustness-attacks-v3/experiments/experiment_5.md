# Experiment 5: Claude Sonnet 4.5 as mutator with 60-token cap

**Hypothesis:** Combining Claude Sonnet 4.5's stronger instruction-following with a hard 60-token cap will produce high-quality, length-controlled distractors — eliminating the length explosion seen in exp2 while retaining Claude's superior mutation quality.

**Status:** SUBMITTED

**Cluster:** oci

**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp5

**Changes:**
- Same as exp2 (Claude Sonnet 4.5 via NVIDIA inference API) but with `--max-distractor-tokens 60`
- Uses **new** few-shot examples (50-60 token target) instead of originals — gives Claude a consistent length signal
- Mutation prompt says "no more than 60 tokens" + tokenizer truncation enforces it
- System prompt for refusal reduction (same as exp2)

**Config:** Qwen3-8B eval target, Claude Sonnet 4.5 mutation model via NVIDIA API, 8 GPUs eval, v3 pipeline, 50 iterations, max-distractor-tokens=60

**Few-shot examples:** new (rewritten, 65/100 in 50-60 token range)

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp5 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 50 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp5 \
    --mutation-api-url https://inference-api.nvidia.com \
    --mutation-api-model azure/anthropic/claude-sonnet-4-5 \
    --benchmark gpqa \
    --subset diamond \
    --max-distractor-tokens 60
```

**Progress:** _not started_

**Results:** _pending_
