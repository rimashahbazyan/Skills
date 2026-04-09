# Experiment 3: Hard token cap on generated distractors

**Hypothesis:** Truncating mutated distractors to 60 Qwen3-8B tokens will eliminate false-positive 0% scores caused by eval model token exhaustion, producing more meaningful accuracy measurements and a cleaner attack signal.

**Changes:**
- After `perflab_mutation_call` returns mutated text in step1.py, tokenize with Qwen3-8B and truncate to 60 tokens if exceeded
- Uses new improved few-shot examples (same as exp1)
- Uses gpt-oss-120b as mutator (same as exp1/baseline)

**Config:** Qwen3-8B, gpt-oss-120b, 8 GPUs, v3 pipeline, 100 iterations

**Status:** SUBMITTED
**Cluster:** oci
**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp3

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp3 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp3 \
    --mutation-model /hf_models/gpt-oss-120b \
    --benchmark gpqa \
    --subset diamond \
    --max-distractor-tokens 60
```

**Progress:** _not started_

**Results:** _pending_
