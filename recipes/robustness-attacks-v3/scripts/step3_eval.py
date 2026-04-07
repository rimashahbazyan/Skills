"""Consolidated step 2 + step 3 (eval) for the robustness-attacks pipeline.

Runs on the cluster inside a container with a vLLM co-process server.
Replaces the previous step2 run_cmd + 15 separate eval() jobs with a single
Slurm job that:
  1. Injects distractors into benchmark files (step 2 logic)
  2. Launches 15 concurrent inference processes against one shared vLLM server
  3. Summarizes results for each (type, position) slot
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

from constants import DISTRACTOR_TYPES
from step2 import ensure_benchmark_file, read_jsonl, validate_and_index_distractors, write_injected_files
from utils import wait_for_local_server


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated eval: inject distractors, run 15 evals against shared vLLM server, summarize."
    )
    parser.add_argument("--output-folder", type=str, required=True, help="Root output directory (cluster path).")
    parser.add_argument("--iteration-id", type=str, required=True, help="Current iteration number.")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g. gpqa).")
    parser.add_argument("--subset", type=str, required=True, help="Benchmark subset (e.g. diamond).")
    parser.add_argument("--model", type=str, required=True, help="Model name for server config.")
    parser.add_argument("--server-port", type=int, default=5000, help="vLLM server port (default: 5000).")
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=512,
        help="Max concurrent requests per eval slot (default: 512).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log/output directory. Defaults to {output_folder}/logs.",
    )
    return parser.parse_args()


def discover_positions() -> list[str]:
    """Discover eval positions from prompt files on the cluster."""
    pattern = "/nemo_run/code/recipes/robustness-attacks-v3/prompts/eval-prompt-*.yaml"
    prompt_files = glob.glob(pattern)
    if not prompt_files:
        raise FileNotFoundError(f"No eval prompt files found matching {pattern}")
    positions = sorted(
        set(
            os.path.splitext(os.path.basename(f))[0].split("eval-prompt-")[1]
            for f in prompt_files
        )
    )
    logging.info("Discovered positions from prompt files: %s", positions)
    return positions


def run_step2_inline(output_folder: str, iteration_id: str, benchmark: str, subset: str) -> None:
    """Run step 2 (distractor injection) inline."""
    workdir = Path(output_folder)
    benchmark_file = ensure_benchmark_file(benchmark, subset, workdir)
    iter_file = workdir / "archive" / f"iter{iteration_id}.jsonl"
    if not iter_file.exists():
        raise FileNotFoundError(f"Distractor file not found: {iter_file}")

    benchmark_rows = read_jsonl(benchmark_file)
    distractor_types, positions, distractor_index = validate_and_index_distractors(iter_file)

    write_injected_files(
        benchmark_rows=benchmark_rows,
        benchmark=benchmark,
        subset=subset,
        iteration=int(iteration_id),
        distractor_index=distractor_index,
        distractor_types=distractor_types,
        positions=positions,
        workdir=workdir,
    )
    logging.info(
        "Step2 (inline): injected %d x %d = %d files.",
        len(distractor_types),
        len(positions),
        len(distractor_types) * len(positions),
    )


def build_generate_cmd(
    *,
    input_file: str,
    output_file: str,
    prompt_config: str,
    model: str,
    server_port: int,
    max_concurrent_requests: int,
) -> list[str]:
    """Build the command for a single inference process."""
    return [
        sys.executable, "-m", "nemo_skills.inference.generate",
        f"++skip_filled=True",
        f"++input_file={input_file}",
        f"++output_file={output_file}",
        f"++prompt_config={prompt_config}",
        f"++server.host=127.0.0.1",
        f"++server.port={server_port}",
        f"++server.model={model}",
        f"++server.server_type=vllm",
        f"++inference.tokens_to_generate=32768",
        f"++max_concurrent_requests={max_concurrent_requests}",
        f"++inference.endpoint_type=text",
        f"++parse_reasoning=True",
        f"++eval_type=multichoice",
    ]


def build_summarize_cmd(*, results_folder: str, benchmark_name: str, metrics_path: str) -> list[str]:
    """Build the command for summarize_results."""
    return [
        sys.executable, "-m", "nemo_skills.pipeline.summarize_results",
        results_folder,
        "--benchmarks", benchmark_name,
        "--save_metrics_path", metrics_path,
        "--metric_type=multichoice",
    ]


def main() -> None:
    args = parse_args()
    output_folder = args.output_folder
    iteration_id = args.iteration_id
    log_dir = args.log_dir or f"{output_folder}/logs"

    # --- Step 2: inject distractors ---
    logging.info("=== Step 2 (inline): injecting distractors ===")
    run_step2_inline(output_folder, iteration_id, args.benchmark, args.subset)

    # --- Discover positions ---
    positions = discover_positions()

    # --- Wait for vLLM server ---
    server_endpoint = f"http://localhost:{args.server_port}/v1"
    wait_for_local_server(server_endpoint)
    logging.info("vLLM server ready at port %d", args.server_port)

    # --- Run 15 evals sequentially against the shared server ---
    # Running sequentially avoids KV cache contention: with tokens_to_generate=32768,
    # 15 concurrent processes would queue ~1920 requests but only ~256 can run at once,
    # making concurrent execution slower than sequential.
    total_slots = len(DISTRACTOR_TYPES) * len(positions)
    logging.info("=== Step 3: running %d evals sequentially against shared server ===", total_slots)

    failures = []
    for slot_idx, (distractor_type, position) in enumerate(
        (dt, pos) for dt in DISTRACTOR_TYPES for pos in positions
    ):
        slot_label = f"type-{distractor_type}_pos-{position}"
        slot_dir = f"{log_dir}/iter{iteration_id}/step3/{slot_label}/eval-results"

        input_file = (
            f"{output_folder}/iter{iteration_id}/"
            f"{args.benchmark}_{args.subset}__type-{distractor_type}__pos-{position}.jsonl"
        )
        output_file = f"{slot_dir}/iter{iteration_id}/output-rs0.jsonl"
        prompt_config = f"/nemo_run/code/recipes/robustness-attacks-v3/prompts/eval-prompt-{position}.yaml"

        cmd = build_generate_cmd(
            input_file=input_file,
            output_file=output_file,
            prompt_config=prompt_config,
            model=args.model,
            server_port=args.server_port,
            max_concurrent_requests=args.max_concurrent_requests,
        )

        logging.info("[%d/%d] Running eval: %s", slot_idx + 1, total_slots, slot_label)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("FAILED: %s (exit code %d)\n%s", slot_label, result.returncode, result.stdout + result.stderr)
            failures.append(slot_label)
        else:
            logging.info("[%d/%d] Completed: %s", slot_idx + 1, total_slots, slot_label)

    if failures:
        raise RuntimeError(
            f"Inference failed for {len(failures)}/{total_slots} slot(s): {failures}"
        )

    logging.info("All %d inference processes completed successfully.", len(processes))

    # --- Summarize results sequentially ---
    logging.info("=== Summarizing results ===")
    for distractor_type in DISTRACTOR_TYPES:
        for position in positions:
            slot_label = f"type-{distractor_type}_pos-{position}"
            slot_dir = f"{log_dir}/iter{iteration_id}/step3/{slot_label}/eval-results"
            metrics_path = f"{slot_dir}/iter{iteration_id}/metrics.json"

            cmd = build_summarize_cmd(
                results_folder=slot_dir,
                benchmark_name=f"iter{iteration_id}",
                metrics_path=metrics_path,
            )

            logging.info("Summarizing: %s", slot_label)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error("Summarize FAILED for %s:\n%s", slot_label, result.stderr)
                raise RuntimeError(f"Summarize failed for {slot_label}")

    logging.info("=== Step 3 eval completed for iteration %s ===", iteration_id)


if __name__ == "__main__":
    main()
