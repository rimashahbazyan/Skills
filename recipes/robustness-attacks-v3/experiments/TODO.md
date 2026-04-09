# Robustness Attacks v3 — TODO

## Active
- [ ] **Relaunch exp1** on oci (iter 1-9 done, needs to resume from iter 10)
- [ ] **Relaunch exp2** on oci (iter 1-8 done, needs to resume from iter 9)

## Backlog
- [ ] **Fix exp2 distractor length explosion** — Claude Sonnet 4.5 generates 850-2111 char distractors causing 88% no_answer in some slots. Options:
  1. Stricter length constraint in mutation prompt (e.g., "under 100 characters")
  2. Use new improved few-shot examples (50-60 tokens) instead of originals
  3. Post-process truncation in step2 (cap at 200 chars)
  - See [experiment_2.md](experiment_2.md) for full analysis

## Done
- [x] Analyze v2 pipeline timing (step3 is 94.5% of GPU time)
- [x] Build v3 pipeline (merge step2→step1, step4→next step1, summarize→step4)
- [x] Test v3 end-to-end on oci (2 iterations, 40 min, all correct)
- [x] Rewrite few-shot examples for 50-60 token target (65/100 in range)
- [x] Force-add JSONL files to git (bypassing *.jsonl gitignore)
- [x] Fix max_tokens 1024→8192 for reasoning model (gpt-oss-120b)
- [x] Fix eval time limit 20min→1hr (CODE_SNIPPET_pos-0 timeout)
- [x] Add eval retry logic + make step4 resilient to failed evals
- [x] Add system prompt to reduce Claude refusals on ENCRYPTED_TEXT mutations
- [x] Add INFERENCE_NVIDIA_KEY to oci cluster config
