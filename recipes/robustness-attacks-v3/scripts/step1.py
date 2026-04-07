import argparse
import json
import os
import random
import uuid
from pathlib import Path
from typing import List, Dict

import yaml
from openai import AzureOpenAI, OpenAI

from constants import INITIAL_DISTRACTORS
from utils import wait_for_local_server

# ==========================
# -------- CONFIG ----------
# ==========================

AZURE_KEY = os.getenv("AZURE_KEY", "")  # Set in environment
MODEL = os.getenv("AZURE_MODEL", "gpt-5-chat-20250807")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://llm-proxy.perflab.nvidia.com")

# Default port NeMo-Run uses when launching a co-process vLLM server via run_cmd.
_LOCAL_SERVER_PORT = 5000

DEFAULT_TEMPERATURE = 1.0  # high enough to ensure varied outputs across iterations
DEFAULT_SEED_BASE = 42     # iteration seed = SEED_BASE + int(iteration_id)

# ==========================
# ----- Azure Client -------
# ==========================

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
)

# Whether to pass `seed` to the chat completions call.  Disabled for external
# APIs (e.g. NVIDIA inference API) where the provider may not support it.
_PASS_SEED = True

# ==========================
# ---- Prompt Loading ------
# ==========================

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_mutation_prompt() -> dict:
    with open(_PROMPTS_DIR / "mutation-prompt.yaml") as f:
        return yaml.safe_load(f)


_MUTATION_PROMPT = _load_mutation_prompt()

# ==========================
# --- Few-shot examples ----
# ==========================

# Directory containing one JSONL file per source type.
# Overridden at runtime via --few-shot-dir.
_FEW_SHOT_DIR = _PROMPTS_DIR / "few-shot-examples"

# Maps internal type names to the display names used in the mutation prompt
# and in the few-shot example files.
_TYPE_DISPLAY = {
    "RANDOM_FACT":    "Random Fact",
    "CODE_SNIPPET":   "Code Snippet",
    "ENCRYPTED_TEXT": "Encrypted Text",
    "MARKUP_NOISE":   "Markup Noise",
    "MATH_FACT":      "Math Fact",
}

# Maps internal type name to the corresponding few-shot JSONL filename.
_FEW_SHOT_FILE = {
    "RANDOM_FACT":    "random_fact.jsonl",
    "CODE_SNIPPET":   "code_snippet.jsonl",
    "ENCRYPTED_TEXT": "encrypted_text.jsonl",
    "MARKUP_NOISE":   "markup_noise.jsonl",
    "MATH_FACT":      "math_fact.jsonl",
}

# Cache: loaded once per source type per process.
_FEW_SHOT_CACHE: Dict[str, List[Dict]] = {}


def _load_few_shot_examples(source_type: str, target_type: str, n: int = 5) -> List[Dict]:
    """Return up to *n* randomly sampled few-shot examples for (source → target)."""
    if source_type not in _FEW_SHOT_CACHE:
        path = _FEW_SHOT_DIR / _FEW_SHOT_FILE[source_type]
        if path.exists():
            with open(path) as f:
                _FEW_SHOT_CACHE[source_type] = [json.loads(l) for l in f if l.strip()]
        else:
            _FEW_SHOT_CACHE[source_type] = []

    target_display = _TYPE_DISPLAY[target_type]
    pool = [e for e in _FEW_SHOT_CACHE[source_type] if e.get("target_type") == target_display]
    return random.sample(pool, min(n, len(pool)))


# ==========================
# ----- File Handling ------
# ==========================

def create_initial_state(state_file: Path):
    """Create state file with initial distractors."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        for item in INITIAL_DISTRACTORS:
            initial_item = item.copy()
            initial_item["parent_id"] = None
            initial_item["parent_score"] = None
            f.write(json.dumps(initial_item) + "\n")
    print(f"Created initial {state_file}")


def load_state(state_file: Path) -> List[Dict]:
    """Load state from JSONL file."""
    with open(state_file, "r") as f:
        return [json.loads(line) for line in f]


def save_state(state: List[Dict], state_file: Path):
    """Save state to JSONL file."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        for item in state:
            f.write(json.dumps(item) + "\n")


# ==========================
# ---- Mutation Logic ------
# ==========================

def _pick_source_for_slot(target_type: str, state: List[Dict], used_ids: set) -> Dict:
    """Pick a random distractor whose type differs from *target_type* and hasn't been used yet.

    No seed is set intentionally — each step1 invocation draws a fresh
    random sequence so source choices vary across iterations.
    Falls back to allowing reuse only if all different-type distractors are exhausted.
    """
    candidates = [d for d in state if d["type"] != target_type and d["id"] not in used_ids]
    if not candidates:
        candidates = [d for d in state if d["type"] != target_type]
    return random.choice(candidates)


def _format_examples(examples: List[Dict], source_display: str, target_display: str) -> str:
    """Render few-shot examples as a block of input/output pairs."""
    lines = ["Here are examples of this transformation:\n"]
    for ex in examples:
        lines.append("---")
        lines.append(f'Source Type: {source_display}')
        lines.append(f'Distractor: "{ex["initial_distractor"]}"')
        lines.append(f'Target Type: {target_display}')
        lines.append("")
        lines.append(f'"{ex["mutated_distractor"]}"')
        lines.append("")
    return "\n".join(lines)


def perflab_mutation_call(original_item: Dict, new_type: str, temperature: float, seed: int) -> str:
    """
    Call the LLM to mutate a distractor into a new type.
    Position remains the same; type changes.
    Prompts are loaded from prompts/mutation-prompt.yaml.
    """
    source_type = original_item["type"]
    source_display = _TYPE_DISPLAY[source_type]
    target_display = _TYPE_DISPLAY[new_type]

    examples = _load_few_shot_examples(source_type, new_type)
    examples_block = _format_examples(examples, source_display, target_display)

    user_prompt = _MUTATION_PROMPT["user_template"].format(
        examples_block=examples_block,
        source_display=source_display,
        distractor=original_item["distractor"],
        target_display=target_display,
    )

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    kwargs = dict(model=MODEL, messages=messages, temperature=temperature)
    if _PASS_SEED:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content.strip()
    # Strip surrounding quotes if the model followed the output format literally.
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return raw


def mutate_state(state: List[Dict], iteration_id: str, temperature: float, seed_base: int) -> List[Dict]:
    """Refresh every grid slot while keeping the full type grid intact.

    For each slot (type=D1, position=P):
      1. Randomly select a *source* distractor of a different type D2 from
         the current state.
      2. Mutate D2's text into D1 style via the LLM.
      3. Keep the slot's type (D1) and position (P) — the grid never shrinks.
      4. Track lineage: parent_id points to the previous D1 occupant so
         step4 can compare the new distractor against the one it replaces.

    Seed formula: seed = seed_base + int(iteration_id). Seeding before the loop
    ensures both source selection (random.choice) and few-shot sampling
    (random.sample) are reproducible for a given iteration.
    """
    seed = seed_base + int(iteration_id)
    random.seed(seed)

    updated_state = []
    used_source_ids: set = set()

    for item in state:
        target_type = item["type"]   # D1: the type slot being refreshed
        position = item["position"]

        source_item = _pick_source_for_slot(target_type, state, used_source_ids)
        used_source_ids.add(source_item["id"])

        print(
            f"Slot ({target_type}, pos={position}): "
            f"source type={source_item['type']} ID={source_item['id']} -> {target_type}"
        )

        new_text = perflab_mutation_call(source_item, target_type, temperature, seed)

        updated_item = item.copy()
        updated_item["parent_id"] = item["id"]
        updated_item["parent_score"] = item.get("eval_score")
        updated_item["source_id"] = source_item["id"]
        updated_item["source_type"] = source_item["type"]
        updated_item["source_distractor"] = source_item["distractor"]
        updated_item["id"] = f"{target_type}_{position}_{uuid.uuid4().hex[:8]}"
        # type and position are unchanged — grid stays full
        updated_item["distractor"] = new_text
        updated_state.append(updated_item)

    return updated_state


# ==========================
# --------- MAIN -----------
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Execute mutation step for robustness-attacks.")
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder to store current_state.jsonl.",
    )
    parser.add_argument(
        "--iteration-id",
        type=str,
        required=True,
        help="Iteration id for logging purposes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM sampling temperature for mutations. Higher values produce more varied outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help="Base value for the iteration seed. Actual seed = seed-base + iteration-id.",
    )
    parser.add_argument(
        "--mutation-model",
        type=str,
        default=None,
        help="Path to local model for mutations (e.g. /hf_models/gpt-oss-120b). "
             "When set, uses a vLLM server started by NeMo-Run instead of Azure.",
    )
    parser.add_argument(
        "--mutation-endpoint",
        type=str,
        default=f"http://localhost:{_LOCAL_SERVER_PORT}/v1",
        help="OpenAI-compatible endpoint for the local mutation server. "
             "Only used when --mutation-model is set.",
    )
    parser.add_argument(
        "--mutation-api-url",
        type=str,
        default=None,
        help="Base URL for an external OpenAI-compatible mutation API "
             "(e.g. https://inference-api.nvidia.com). "
             "Takes precedence over Azure when set.",
    )
    parser.add_argument(
        "--mutation-api-model",
        type=str,
        default=None,
        help="Model name for the external mutation API "
             "(e.g. azure/anthropic/claude-sonnet-4-5). "
             "Required when --mutation-api-url is set.",
    )
    parser.add_argument(
        "--mutation-api-key-env",
        type=str,
        default="INFERENCE_NVIDIA_KEY",
        help="Environment variable containing the API key for the external mutation API. "
             "Default: NVIDIA_API_KEY.",
    )
    parser.add_argument(
        "--few-shot-dir",
        type=str,
        default=None,
        help="Path to directory containing few-shot example JSONL files. "
             "Defaults to prompts/few-shot-examples/ next to the recipe.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark name (e.g. gpqa). When set, runs step2 injection inline after mutation.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Benchmark subset (e.g. diamond). Required when --benchmark is set.",
    )
    parser.add_argument(
        "--prev-iteration-id",
        type=str,
        default=None,
        help="Previous iteration id. When set, runs step4 (summarize + selection) for the "
             "previous iteration before starting mutation. Eliminates a separate Slurm job.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["benchmark", "llm-judge"],
        default="benchmark",
        help="Evaluation mode passed to step4. Only used when --prev-iteration-id is set.",
    )
    args = parser.parse_args()

    # Override few-shot examples directory if specified.
    global _FEW_SHOT_DIR
    if args.few_shot_dir:
        _FEW_SHOT_DIR = Path(args.few_shot_dir)
        print(f"Using few-shot examples from: {_FEW_SHOT_DIR}")

    # If a local model is requested, wait for the co-process vLLM server
    # (started by NeMo-Run alongside this job) and swap out the Azure client.
    global client, MODEL, _PASS_SEED
    if args.mutation_model:
        wait_for_local_server(args.mutation_endpoint)
        client = OpenAI(base_url=args.mutation_endpoint, api_key="EMPTY")
        MODEL = args.mutation_model
        print(f"Using local model: {args.mutation_model} @ {args.mutation_endpoint}")
    elif args.mutation_api_url:
        api_key = os.getenv(args.mutation_api_key_env, "")
        client = OpenAI(base_url=args.mutation_api_url, api_key=api_key)
        MODEL = args.mutation_api_model
        _PASS_SEED = False  # external APIs may not support seed
        print(f"Using external API model: {MODEL} @ {args.mutation_api_url}")
    else:
        print(f"Using Azure model: {MODEL} @ {AZURE_ENDPOINT}")

    # Run step4 for the previous iteration (summarize + selection) before mutating.
    # This updates current_state.jsonl which step1 then reads for mutation.
    if args.prev_iteration_id:
        print(f"Running step4 for previous iteration {args.prev_iteration_id}...")
        import subprocess, sys
        step4_cmd = [
            sys.executable, "/nemo_run/code/recipes/robustness-attacks-v3/scripts/step4.py",
            "--output-folder", args.output_folder,
            "--iteration-id", args.prev_iteration_id,
            "--eval-mode", args.eval_mode,
        ]
        result = subprocess.run(step4_cmd, cwd="/nemo_run/code/recipes/robustness-attacks-v3/scripts")
        if result.returncode != 0:
            raise RuntimeError(f"Step4 for iteration {args.prev_iteration_id} failed (exit code {result.returncode})")
        print(f"Step4 for iteration {args.prev_iteration_id} completed.")

    state_file = Path(args.output_folder) / "current_state.jsonl"

    print(f"Starting Step1 - Iteration {args.iteration_id}")

    if not state_file.exists():
        create_initial_state(state_file)

    state = load_state(state_file)

    archive_file = Path(args.output_folder) / "archive" / f"iter{args.iteration_id}.jsonl"

    state = mutate_state(state, args.iteration_id, args.temperature, args.seed_base)

    save_state(state, archive_file)

    # Create iter{N}/__init__.py so the eval pipeline can resolve the benchmark config.
    iter_dir = Path(args.output_folder) / f"iter{args.iteration_id}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    init_file = iter_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            'METRICS_TYPE = "multichoice"\n'
            'EVAL_SPLIT = "test"\n'
            'GENERATION_ARGS = "++eval_type=multichoice"\n'
        )
        print(f"Created {init_file}")

    # Optionally run step2 (distractor injection) inline to avoid a separate Slurm job.
    if args.benchmark:
        from step2 import ensure_benchmark_file, read_jsonl as read_jsonl_step2, validate_and_index_distractors, write_injected_files

        workdir = Path(args.output_folder)
        benchmark_file = ensure_benchmark_file(args.benchmark, args.subset, workdir)
        benchmark_rows = read_jsonl_step2(benchmark_file)
        distractor_types, positions, distractor_index = validate_and_index_distractors(archive_file)
        write_injected_files(
            benchmark_rows=benchmark_rows,
            benchmark=args.benchmark,
            subset=args.subset,
            iteration=int(args.iteration_id),
            distractor_index=distractor_index,
            distractor_types=distractor_types,
            positions=positions,
            workdir=workdir,
        )
        print(f"Step2 (inline): injected {len(distractor_types)} x {len(positions)} = {len(distractor_types) * len(positions)} files.")

    print(f"Step1 completed for iteration {args.iteration_id}")


if __name__ == "__main__":
    main()
