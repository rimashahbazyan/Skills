# Experiment 6: Parent-aware mutation prompt

**Hypothesis:** Providing the current slot's parent distractor in the mutation prompt will let the LLM build on what already worked, producing incrementally stronger distractors rather than starting from scratch each iteration. This should lead to faster accuracy degradation.

**Status:** NOT SUBMITTED

**Cluster:** oci

**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp6

**Changes:**
- New mutation prompt: `mutation-prompt-with-parent.yaml` — includes the parent distractor text with instruction to improve upon it
- Added `--mutation-prompt` arg to step1.py and launch_attack.py
- Parent distractor passed through from `mutate_state` → `perflab_mutation_call` → prompt template
- First iteration gets "N/A (first iteration)" as parent since there's no predecessor

**Config:** Qwen3-8B eval target, gpt-oss-120b mutation model, 8 GPUs eval, v3 pipeline, 100 iterations, max-distractor-tokens=60

**Few-shot examples:** new (rewritten, 65/100 in 50-60 token range)

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /workspace/robustness-attacks/gpqa-diamond-qwen8b-v3-exp6 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp6 \
    --mutation-model /hf_models/gpt-oss-120b \
    --benchmark gpqa \
    --subset diamond \
    --max-distractor-tokens 60 \
    --mutation-prompt mutation-prompt-with-parent.yaml
```

**Progress:** _not started_

**Results:** _pending_
