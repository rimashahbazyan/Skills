# Experiment 4: Gemma 4 31B as mutator

**Hypothesis:** Gemma 4 31B-IT may produce different-quality distractor mutations compared to gpt-oss-120b, potentially with better instruction following on length constraints and type transformations.

**Status:** RUNNING

**Cluster:** oci
> **Note:** Using `oci-gemma` cluster config with custom vllm image (`/lustre/fsw/portfolios/llmservice/users/rshahbazyan/images/nemo-skills-vllm-gemma.sqsh`)

**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp4

**Changes:**
- Uses `oci-gemma` cluster config with vllm image: `/lustre/fsw/portfolios/llmservice/users/rshahbazyan/images/nemo-skills-vllm-gemma.sqsh`
- Mutation model: `/hf_models/gemma-4-31b-it` (8 GPUs)
- Uses new improved few-shot examples (same as exp1/exp3)
- Includes `--max-distractor-tokens 60` to prevent length explosion

**Config:** Qwen3-8B eval target, gemma-4-31b-it mutation model, 8 GPUs eval, 8 GPUs mutation, v3 pipeline, 100 iterations, max-distractor-tokens=60

**Few-shot examples:** new (rewritten, 65/100 in 50-60 token range)

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci-gemma \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp4 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp4 \
    --mutation-model /hf_models/gemma-4-31b-it \
    --mutation-server-gpus 8 \
    --benchmark gpqa \
    --subset diamond \
    --max-distractor-tokens 60
```

**Progress:** _not started_

**Results:** _pending_
