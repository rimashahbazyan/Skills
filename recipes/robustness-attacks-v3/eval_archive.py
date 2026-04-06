"""Benchmark eval on archived distractors for every Nth completed iteration.

Does NOT run step1 (mutation) or step4 (selection). Reads the existing archive
and runs step2 (inject) + 15 parallel ns eval jobs (step3) per selected iteration.

Useful for evaluating llm-judge runs (where step3 benchmark eval was never run)
or for re-evaluating benchmark runs with updated model configs.

Usage:
    python recipes/robustness-attacks-v3/eval_archive.py \\
        --cluster ord \\
        --output-folder /workspace/robustness-attacks/my-run \\
        --model-name /hf_models/Qwen3-8B \\
        --iter-num 34 \\
        --n 5 \\
        --expname-prefix robustness-beval-my-run
"""

import argparse
import glob
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import eval, run_cmd, wrap_arguments
from nemo_skills.utils import get_logger_name, setup_logging
from scripts.constants import DISTRACTOR_TYPES


LOG = logging.getLogger(get_logger_name(__file__))


_BENCHMARK_INIT_CONTENT = (
    'METRICS_TYPE = "multichoice"\n'
    'EVAL_SPLIT = "test"\n'
    'GENERATION_ARGS = "++eval_type=multichoice"\n'
)


def _upload_iter_stubs(
    cluster_config: dict,
    output_folder: str,
    iteration: int,
    benchmark: str,
    subset: str,
    distractor_types: list,
    positions: list,
) -> None:
    """Create iter{N}/__init__.py and empty stub data files on the cluster."""
    real_iter_dir = pipeline_utils.get_unmounted_path(cluster_config, f"{output_folder}/iter{iteration}")

    if cluster_config.get("executor") == "local":
        iter_dir = Path(real_iter_dir)
        iter_dir.mkdir(parents=True, exist_ok=True)

        init_file = iter_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(_BENCHMARK_INIT_CONTENT)
            LOG.info("Created benchmark init file at %s", init_file)

        for distractor_type in distractor_types:
            for position in positions:
                filename = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}.jsonl"
                stub_file = iter_dir / filename
                if not stub_file.exists():
                    stub_file.write_text("")
                    LOG.info("Created stub data file at %s", stub_file)
        return

    tunnel = pipeline_utils.get_tunnel(cluster_config)
    tunnel.run(f"mkdir -p {real_iter_dir}", hide=True)

    init_path = f"{output_folder}/iter{iteration}/__init__.py"
    if not pipeline_utils.cluster_path_exists(cluster_config, init_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix="__init__.py", delete=False) as f:
            f.write(_BENCHMARK_INIT_CONTENT)
            tmp_path = f.name
        pipeline_utils.cluster_upload(cluster_config, tmp_path, f"{real_iter_dir}/__init__.py", verbose=False)
        os.unlink(tmp_path)
        LOG.info("Uploaded benchmark init file to %s", init_path)

    for distractor_type in distractor_types:
        for position in positions:
            filename = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}.jsonl"
            stub_path = f"{output_folder}/iter{iteration}/{filename}"
            if not pipeline_utils.cluster_path_exists(cluster_config, stub_path):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    tmp_path = f.name
                pipeline_utils.cluster_upload(cluster_config, tmp_path, f"{real_iter_dir}/{filename}", verbose=False)
                os.unlink(tmp_path)
                LOG.info("Uploaded stub data file to %s", stub_path)


def _iteration_complete(cluster_config: dict, output_folder: str, iteration: int) -> bool:
    """Return True if the archive for this iteration has all entries scored."""
    remote_path = pipeline_utils.get_unmounted_path(
        cluster_config, f"{output_folder}/archive/iter{iteration}.jsonl"
    )

    if not pipeline_utils.cluster_path_exists(cluster_config, remote_path):
        return False

    tunnel = pipeline_utils.get_tunnel(cluster_config)
    result = tunnel.run(
        f'total=$(grep -c . {remote_path}); '
        f'scored=$(grep -c "eval_score" {remote_path}); '
        f'echo "$total $scored"',
        hide=True,
        warn=True,
    )
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return False
    total, scored = int(parts[0]), int(parts[1])
    return total > 0 and total == scored


def _eval_complete(
    cluster_config: dict,
    output_folder: str,
    iteration: int,
    distractor_types: list,
    positions: list,
) -> bool:
    """Return True if all metrics.json files already exist for this iteration."""
    for distractor_type in distractor_types:
        for position in positions:
            metrics_path = (
                f"{output_folder}/logs/iter{iteration}/step3"
                f"/type-{distractor_type}_pos-{position}"
                f"/eval-results/iter{iteration}/metrics.json"
            )
            if not pipeline_utils.cluster_path_exists(cluster_config, metrics_path):
                return False
    return True


def _step2_complete(
    cluster_config: dict,
    output_folder: str,
    iteration: int,
    benchmark: str,
    subset: str,
    distractor_types: list,
    positions: list,
) -> bool:
    """Return True if all injected JSONL files exist and are non-empty."""
    for distractor_type in distractor_types:
        for position in positions:
            filename = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}.jsonl"
            file_path = f"{output_folder}/iter{iteration}/{filename}"
            remote_path = pipeline_utils.get_unmounted_path(cluster_config, file_path)

            if not pipeline_utils.cluster_path_exists(cluster_config, remote_path):
                return False

            # Check file is non-empty
            tunnel = pipeline_utils.get_tunnel(cluster_config)
            result = tunnel.run(f"wc -l < {remote_path}", hide=True, warn=True)
            line_count = result.stdout.strip()
            if not line_count.isdigit() or int(line_count) == 0:
                return False
    return True


def schedule_eval_iter(
    *,
    cluster: str,
    cluster_config: dict,
    output_folder: str,
    model_name: str,
    iteration: int,
    expname_prefix: str,
    log_dir: str,
    container: str,
    benchmark: str,
    subset: str,
    server_gpus: int = 4,
    max_model_len: int = 40960,
) -> Any:
    """Schedule step2 (if needed) + 15 eval jobs for one iteration. Returns last eval experiment."""
    LOG.info("Scheduling benchmark eval for iteration %d", iteration)

    pattern = "recipes/robustness-attacks-v3/prompts/eval-prompt-*.yaml"
    prompt_files = glob.glob(pattern)
    positions = sorted(
        set(
            os.path.splitext(os.path.basename(p))[0].split("eval-prompt-")[1]
            for p in prompt_files
        )
    )

    _upload_iter_stubs(cluster_config, output_folder, iteration, benchmark, subset, DISTRACTOR_TYPES, positions)

    step2_expname = f"{expname_prefix}_iter{iteration}_step2"
    step2_scheduled = False

    if not _step2_complete(cluster_config, output_folder, iteration, benchmark, subset, DISTRACTOR_TYPES, positions):
        LOG.info("iter%d: step2 injected files missing — scheduling step2", iteration)
        step2_cmd = (
            "python recipes/robustness-attacks-v3/scripts/step2.py "
            f"--iter {iteration} "
            f"--benchmark {benchmark} "
            f"--subset {subset} "
            f"--output-folder {output_folder} "
        )
        run_cmd(
            ctx=wrap_arguments(step2_cmd),
            cluster=cluster,
            container=container,
            expname=step2_expname,
            log_dir=f"{log_dir}/iter{iteration}/step2",
            run_after=[],
            partition=cluster_config.get("cpu_partition"),
        )
        step2_scheduled = True
    else:
        LOG.info("iter%d: step2 files already present — skipping step2", iteration)

    last_exp = None
    for distractor_type in DISTRACTOR_TYPES:
        for position in positions:
            split = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}"
            prompt_template = f"/nemo_run/code/recipes/robustness-attacks-v3/prompts/eval-prompt-{position}.yaml"
            eval_expname = f"{expname_prefix}_iter{iteration}_eval_type-{distractor_type}_pos-{position}"

            last_exp = eval(
                ctx=wrap_arguments(
                    f"++inference.tokens_to_generate={32768} "
                    f"++max_concurrent_requests=512 "
                    f"++inference.endpoint_type=text "
                    f"++skip_filled=True "
                    f"++parse_reasoning=True "
                    f"++prompt_config={prompt_template} "
                ),
                cluster=cluster,
                expname=eval_expname,
                output_dir=f"{log_dir}/iter{iteration}/step3/type-{distractor_type}_pos-{position}",
                log_dir=f"{log_dir}/iter{iteration}/step3/type-{distractor_type}_pos-{position}/logs",
                model=model_name,
                server_type="vllm",
                server_gpus=server_gpus,
                server_args=f"--max-model-len {max_model_len}",
                benchmarks=f"iter{iteration}:{1}",
                num_chunks=1,
                split=split,
                data_dir=f"{output_folder}",
                run_after=[step2_expname] if step2_scheduled else [],
            )

    return last_exp


def run_archive_eval(
    *,
    cluster: str,
    output_folder: str,
    model_name: str,
    selected_iters: List[int],
    expname_prefix: str,
    log_dir: str,
    container: str,
    benchmark: str,
    subset: str,
    server_gpus: int = 4,
    max_model_len: int = 40960,
    iter_batch_size: int = 10,
) -> None:
    """Submit eval jobs in batches of iter_batch_size, waiting between batches."""
    setup_logging(disable_hydra_logs=False, use_rich=True)
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir=None)

    pattern = "recipes/robustness-attacks-v3/prompts/eval-prompt-*.yaml"
    positions = sorted(
        set(
            os.path.splitext(os.path.basename(p))[0].split("eval-prompt-")[1]
            for p in glob.glob(pattern)
        )
    )

    LOG.info("Selected %d iterations for benchmark eval: %s", len(selected_iters), selected_iters)

    for batch_start_idx in range(0, len(selected_iters), iter_batch_size):
        batch = selected_iters[batch_start_idx: batch_start_idx + iter_batch_size]
        LOG.info("Submitting eval batch: iterations %s", batch)

        last_exp = None
        for iteration in batch:
            if _eval_complete(cluster_config, output_folder, iteration, DISTRACTOR_TYPES, positions):
                LOG.info("iter%d: all metrics.json already exist — skipping", iteration)
                continue

            last_exp = schedule_eval_iter(
                cluster=cluster,
                cluster_config=cluster_config,
                output_folder=output_folder,
                model_name=model_name,
                iteration=iteration,
                expname_prefix=expname_prefix,
                log_dir=log_dir,
                container=container,
                benchmark=benchmark,
                subset=subset,
                server_gpus=server_gpus,
                max_model_len=max_model_len,
            )

        if last_exp is not None:
            LOG.info("Waiting for eval batch %s to complete...", batch)
            last_exp._wait_for_jobs(last_exp.jobs)
            LOG.info("Eval batch %s completed", batch)
        else:
            LOG.info("Eval batch %s — all iterations already evaluated", batch)

    LOG.info("Archive eval complete for %d iterations.", len(selected_iters))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark eval on archived distractors for every Nth iteration."
    )
    parser.add_argument("--cluster", type=str, default="local", help="Cluster config name.")
    parser.add_argument("--output-folder", type=str, required=True, help="Root output folder (same as attack run).")
    parser.add_argument("--model-name", type=str, required=True, help="Model to evaluate.")
    parser.add_argument(
        "--expname-prefix",
        type=str,
        default="robustness-beval",
        help="Prefix for eval job names. Use a different prefix than the attack run to avoid conflicts.",
    )
    parser.add_argument("--container", type=str, default="nemo-skills", help="Container image.")
    parser.add_argument("--benchmark", type=str, default="gpqa", help="Benchmark name.")
    parser.add_argument("--subset", type=str, default="diamond", help="Benchmark subset name.")
    parser.add_argument("--server-gpus", type=int, default=4, help="GPUs for vLLM eval server.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="Max token context length for the vLLM server.",
    )
    parser.add_argument("--iter-num", type=int, required=True, help="Total iterations to consider (1 to iter-num).")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Evaluate every Nth completed iteration. Default 1 = all iterations.",
    )
    parser.add_argument(
        "--iter-batch-size",
        type=int,
        default=10,
        help="Number of iterations to submit per batch before waiting for completion.",
    )

    args = parser.parse_args()

    cluster_config = pipeline_utils.get_cluster_config(args.cluster, config_dir=None)

    LOG.info("Scanning archive for completed iterations (1 to %d)...", args.iter_num)
    completed = [
        i for i in range(1, args.iter_num + 1)
        if _iteration_complete(cluster_config, args.output_folder, i)
    ]
    LOG.info("Found %d completed iterations: %s", len(completed), completed)

    selected = completed[:: args.n]
    LOG.info("After applying --n=%d: %d iterations selected: %s", args.n, len(selected), selected)

    if not selected:
        LOG.warning("No completed iterations found — nothing to evaluate.")
        return

    run_archive_eval(
        cluster=args.cluster,
        output_folder=args.output_folder,
        model_name=args.model_name,
        selected_iters=selected,
        expname_prefix=args.expname_prefix,
        log_dir=f"{args.output_folder}/logs",
        container=args.container,
        benchmark=args.benchmark,
        subset=args.subset,
        server_gpus=args.server_gpus,
        max_model_len=args.max_model_len,
        iter_batch_size=args.iter_batch_size,
    )


if __name__ == "__main__":
    main()
