# Experiment 1: Improved few-shot examples (50-60 token target)

**Hypothesis:** Current few-shot examples have wildly inconsistent token lengths (16-514 tokens, only 16% in 50-60 range). Rewriting all 100 examples to consistently hit 50-60 tokens will produce more uniform distractor lengths and potentially stronger attacks.

**Changes:**
- Rewrote all 100 few-shot examples across 5 files (code_snippet, encrypted_text, markup_noise, math_fact, random_fact)
- 65/100 mutated_distractors verified to be 50-60 Qwen3-8B tokens (up from 16/100)
- Eliminated duplicate content (e.g., repeated octopus facts)
- Ensured quality: each mutation matches target type, is coherent, and is a plausible distractor

**Config:** Qwen3-8B, gpt-oss-120b, 8 GPUs, v3 pipeline, 100 iterations

**Status:** RUNNING (relaunched after OCI login node crash)
**Cluster:** oci
**Output path:** /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp1

**Launch command:**
```bash
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 python recipes/robustness-attacks-v3/launch_attack.py \
    --cluster oci \
    --output-folder /lustre/fsw/portfolios/llmservice/users/rshahbazyan/robustness-attacks/gpqa-diamond-qwen8b-v3-exp1 \
    --model-name /hf_models/Qwen3-8B \
    --iter-num 100 \
    --iter-batch-size 5 \
    --expname-prefix robustness-v3-exp1 \
    --mutation-model /hf_models/gpt-oss-120b \
    --benchmark gpqa \
    --subset diamond
```

**Progress:** iter 1-9 scored, relaunched to continue from iter 10

**Results:** _pending_

---

## Issue: Distractor length explosion despite improved few-shot examples

**Observed in iter 5:** gpt-oss-120b ignores the 50-60 token few-shot signal and generates distractors up to 549 tokens. Long distractors cause Qwen3-8B to exhaust its 32K token budget on reasoning without producing an answer (`no_answer` up to 95%).

**Iter 5 distractor token counts vs eval scores:**

| Slot | Tokens | Score | Notes |
|------|--------|-------|-------|
| RANDOM_FACT pos=0 | 56 | 61.6% | Normal — matches few-shot target |
| RANDOM_FACT pos=1 | 33 | 63.1% | Normal |
| RANDOM_FACT pos=2 | 181 | 61.6% | Overlong but score OK |
| CODE_SNIPPET pos=0 | 247 | **0.0%** | 95% no_answer, finish_reason=length |
| CODE_SNIPPET pos=1 | 281 | 50.5% | |
| CODE_SNIPPET pos=2 | 278 | 54.5% | |
| ENCRYPTED_TEXT pos=0 | 279 | **12.1%** | High no_answer |
| ENCRYPTED_TEXT pos=1 | 272 | 61.1% | |
| ENCRYPTED_TEXT pos=2 | 266 | 61.1% | |
| MARKUP_NOISE pos=0 | 107 | 59.6% | |
| MARKUP_NOISE pos=1 | 472 | 55.1% | |
| MARKUP_NOISE pos=2 | 144 | 60.1% | |
| MATH_FACT pos=0 | 549 | **9.1%** | High no_answer |
| MATH_FACT pos=1 | 74 | 59.1% | Normal |
| MATH_FACT pos=2 | 303 | 64.1% | |

**Pattern:** Distractors above ~250 tokens cause severe `no_answer` rates. The model generates 32K tokens of reasoning (hitting `finish_reason: length`) without ever completing `</think>`. Step4 treats the resulting 0% score as a "successful attack" (lower than parent), keeping these broken distractors — a false positive.

**Same root cause as exp2** — the mutation prompt says "no more than 500 tokens" but this is too generous. Baseline distractors were ~15-20 tokens. The few-shot examples target 50-60 but gpt-oss-120b doesn't follow them consistently for CODE_SNIPPET, ENCRYPTED_TEXT, and MATH_FACT types.
