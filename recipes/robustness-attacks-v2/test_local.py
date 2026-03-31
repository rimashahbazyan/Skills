"""Local test harness for the robustness-attacks recipe.

Bypasses Azure OpenAI (step1 mutation) by writing initial distractors directly,
then drives step2 through its full logic, and verifies the position alignment and
the local-executor stub creation that were previously broken.

Run from the repo root:
    python recipes/robustness-attacks/test_local.py --output-dir /tmp/robustness-test
"""

import argparse
import glob as glob_mod
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "recipes" / "robustness-attacks" / "scripts"))

from scripts.constants import DISTRACTOR_TYPES, INITIAL_DISTRACTORS  # noqa: E402

BENCHMARK = "mini_mcq"
SUBSET = "test"
ITERATION = 1


def write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def setup_initial_state(output_dir: Path) -> None:
    """Write current_state.jsonl and archive/iter1.jsonl directly (bypasses Azure)."""
    print("\n=== SETUP: writing initial distractor state (no Azure call) ===")
    state = [dict(d, parent_id=None, parent_score=None) for d in INITIAL_DISTRACTORS]
    write_jsonl(output_dir / "current_state.jsonl", state)
    write_jsonl(output_dir / "archive" / f"iter{ITERATION}.jsonl", state)
    positions = sorted(set(d["position"] for d in INITIAL_DISTRACTORS))
    print(f"  Distractor positions: {positions}  (types: {DISTRACTOR_TYPES})")


def run_step2(output_dir: Path) -> None:
    """Run step2.py as a subprocess (mirrors what run_cmd does inside container)."""
    print("\n=== STEP 2: injecting distractors into benchmark ===")
    local_benchmark_src = REPO_ROOT / "nemo_skills" / "dataset" / BENCHMARK / f"{SUBSET}.jsonl"
    local_benchmark_dst = output_dir / f"{BENCHMARK}_{SUBSET}.jsonl"
    if not local_benchmark_dst.exists():
        shutil.copy2(local_benchmark_src, local_benchmark_dst)
        print(f"  Copied benchmark: {local_benchmark_src.name} -> {local_benchmark_dst}")

    step2 = REPO_ROOT / "recipes" / "robustness-attacks" / "scripts" / "step2.py"
    cmd = [
        sys.executable, str(step2),
        "--iter", str(ITERATION),
        "--benchmark", BENCHMARK,
        "--subset", SUBSET,
        "--output-folder", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})\n{result.stderr}")
        return

    iter_dir = output_dir / f"iter{ITERATION}"
    files = sorted(iter_dir.glob("*.jsonl")) if iter_dir.exists() else []
    print(f"  Files written by step2 ({len(files)} total):")
    for f in files:
        lines = sum(1 for _ in open(f))
        print(f"    {f.name}  ({lines} rows)")


def check_position_alignment(output_dir: Path) -> bool:
    """Verify positions in archive match prompt file suffixes."""
    print("\n=== Position alignment check ===")
    archive = output_dir / "archive" / f"iter{ITERATION}.jsonl"
    archive_positions = sorted(set(
        str(json.loads(l)["position"]) for l in open(archive)
    ))
    prompt_pattern = str(REPO_ROOT / "recipes" / "robustness-attacks" / "prompts" / "eval-prompt-*.yaml")
    prompt_positions = sorted(set(
        os.path.splitext(os.path.basename(p))[0].split("eval-prompt-")[1]
        for p in glob_mod.glob(prompt_pattern)
    ))
    print(f"  Archive positions:  {archive_positions}")
    print(f"  Prompt positions:   {prompt_positions}")
    if set(archive_positions) == set(prompt_positions):
        print("  OK — positions match.")
        return True
    print("  MISMATCH — fix constants.py INITIAL_DISTRACTORS positions to match prompt file suffixes.")
    return False


def check_upload_stubs(output_dir: Path) -> bool:
    """Verify _upload_iter_stubs works for local executor."""
    print("\n=== _upload_iter_stubs on local cluster ===")
    import importlib.util
    from nemo_skills.pipeline import utils as pipeline_utils

    spec = importlib.util.spec_from_file_location(
        "launch_attack",
        REPO_ROOT / "recipes" / "robustness-attacks" / "launch_attack.py",
    )
    launch_attack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_attack)
    _upload_iter_stubs = launch_attack._upload_iter_stubs
    DT = launch_attack.DISTRACTOR_TYPES

    cluster_config = pipeline_utils.get_cluster_config("test-local", config_dir=None)
    prompt_pattern = str(REPO_ROOT / "recipes" / "robustness-attacks" / "prompts" / "eval-prompt-*.yaml")
    positions = sorted(set(
        os.path.splitext(os.path.basename(p))[0].split("eval-prompt-")[1]
        for p in glob_mod.glob(prompt_pattern)
    ))

    # Remove iter dir so stubs are actually created
    iter_dir = output_dir / f"iter{ITERATION}"
    if iter_dir.exists():
        shutil.rmtree(iter_dir)

    try:
        _upload_iter_stubs(cluster_config, str(output_dir), ITERATION, BENCHMARK, SUBSET, DT, positions)
        stubs = sorted(iter_dir.glob("*"))
        print(f"  OK — created {len(stubs)} files under iter{ITERATION}/:")
        for s in stubs:
            print(f"    {s.name}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/robustness-test")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    setup_initial_state(output_dir)
    run_step2(output_dir)
    pos_ok = check_position_alignment(output_dir)
    stubs_ok = check_upload_stubs(output_dir)

    print("\n=== Result ===")
    print(f"  Position alignment: {'PASS' if pos_ok else 'FAIL'}")
    print(f"  Local stub creation: {'PASS' if stubs_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
