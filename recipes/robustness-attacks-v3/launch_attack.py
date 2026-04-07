import argparse
import glob
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

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
	"""Create iter{N}/__init__.py and empty stub data files on the cluster.

	eval() fetches __init__.py to resolve the benchmark config and also checks
	that each data JSONL file exists at submission time (before step2 runs).
	The stub JSONL files are empty placeholders; step2 overwrites them at runtime.
	"""
	real_iter_dir = pipeline_utils.get_unmounted_path(cluster_config, f"{output_folder}/iter{iteration}")

	if cluster_config.get("executor") == "local":
		# For local executor LocalTunnel has no SFTP session; write files directly.
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

	# Upload __init__.py if not already present.
	init_path = f"{output_folder}/iter{iteration}/__init__.py"
	if not pipeline_utils.cluster_path_exists(cluster_config, init_path):
		with tempfile.NamedTemporaryFile(mode="w", suffix="__init__.py", delete=False) as f:
			f.write(_BENCHMARK_INIT_CONTENT)
			tmp_path = f.name
		pipeline_utils.cluster_upload(cluster_config, tmp_path, f"{real_iter_dir}/__init__.py", verbose=False)
		os.unlink(tmp_path)
		LOG.info("Uploaded benchmark init file to %s", init_path)

	# Create empty stub JSONL files so eval() existence check passes before step2 runs.
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
	"""Return True if iteration already finished (all archive entries have eval_score).

	Checks directly on the cluster (no file download) by counting lines
	with and without eval_score in the archive JSONL.
	"""
	remote_path = pipeline_utils.get_unmounted_path(
		cluster_config, f"{output_folder}/archive/iter{iteration}.jsonl"
	)

	if not pipeline_utils.cluster_path_exists(cluster_config, remote_path):
		return False

	# Count total non-empty lines and lines containing "eval_score" on the server.
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


_MUTATION_SERVER_PORT = 5000  # default port NeMo-Run uses for a co-process vLLM server


def schedule_iteration(
	*,
	cluster: str,
	cluster_config: dict,
	output_folder: str,
	model_name: str,
	current_iteration: int,
	prev_iteration: Optional[int] = None,
	expname_prefix: str,
	log_dir: str,
	container: str,
	benchmark: str,
	subset: str,
	server_gpus: int = 1,
	max_model_len: int = 40960,
	mutation_model: Optional[str] = None,
	mutation_server_gpus: int = 8,
	prev_eval_expnames: Optional[list] = None,
	temperature: float = 1.0,
	seed_base: int = 42,
	eval_mode: str = "benchmark",
	judge_model: Optional[str] = None,
	judge_server_gpus: int = 8,
	judge_server_nodes: int = 1,
	judge_max_model_len: int = 16384,
) -> list:
	"""Schedule step1 (with inline step4 for prev iter) + evals.

	Returns list of eval experiment names so the next iteration can depend on them.
	"""
	iteration_expname = f"{expname_prefix}_iter{current_iteration}"
	step1_expname = f"{iteration_expname}_step1"

	LOG.info(f"Scheduling iteration {current_iteration}")

	# step 1 — runs step4(N-1) inline, then mutation, then step2 injection
	step1_cmd = (
		"python recipes/robustness-attacks-v3/scripts/step1.py "
		f"--output-folder {output_folder} "
		f"--iteration-id {current_iteration} "
		f"--temperature {temperature} "
		f"--seed-base {seed_base} "
		f"--benchmark {benchmark} "
		f"--subset {subset}"
	)
	if prev_iteration is not None:
		step1_cmd += f" --prev-iteration-id {prev_iteration} --eval-mode {eval_mode}"
	if mutation_model:
		step1_cmd += (
			f" --mutation-model {mutation_model}"
			f" --mutation-endpoint http://localhost:{_MUTATION_SERVER_PORT}/v1"
		)

	step1_kwargs = dict(
		ctx=wrap_arguments(step1_cmd),
		cluster=cluster,
		container=container,
		expname=step1_expname,
		log_dir=f"{log_dir}/iter{current_iteration}/step1",
		run_after=prev_eval_expnames or [],
	)
	if mutation_model:
		step1_kwargs["server_type"] = "vllm"
		step1_kwargs["server_gpus"] = mutation_server_gpus
		step1_kwargs["model"] = mutation_model
		step1_kwargs["partition"] = cluster_config.get("partition")
		step1_kwargs["sbatch_kwargs"] = '{"time": "00:30:00"}'
	else:
		step1_kwargs["partition"] = cluster_config.get("cpu_partition")
	run_cmd(**step1_kwargs)

	eval_expnames = []

	if eval_mode == "benchmark":
		pattern = "recipes/robustness-attacks-v3/prompts/eval-prompt-*.yaml"
		prompt_files = glob.glob(pattern)
		positions = sorted(
			set(
				os.path.splitext(os.path.basename(file_path))[0].split("eval-prompt-")[1]
				for file_path in prompt_files
			)
		)

		_upload_iter_stubs(cluster_config, output_folder, current_iteration, benchmark, subset, DISTRACTOR_TYPES, positions)

		for distractor_type in DISTRACTOR_TYPES:
			for position in positions:
				split = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}"
				prompt_template = f'/nemo_run/code/recipes/robustness-attacks-v3/prompts/eval-prompt-{position}.yaml'
				eval_expname = f"{iteration_expname}_eval_type-{distractor_type}_pos-{position}"
				eval_expnames.append(eval_expname)

				eval(
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
					output_dir=f"{log_dir}/iter{current_iteration}/step3/type-{distractor_type}_pos-{position}",
					log_dir=f"{log_dir}/iter{current_iteration}/step3/type-{distractor_type}_pos-{position}/logs",
					model=model_name,
					server_type="vllm",
					server_gpus=server_gpus,
					server_args=f"--max-model-len {max_model_len}",
					benchmarks=f"iter{current_iteration}:{1}",
					num_chunks=1,
					split=split,
					data_dir=f"{output_folder}",
					run_after=[step1_expname],
					auto_summarize_results=False,
					sbatch_kwargs={"time": "00:30:00"},
				)

	else:  # llm-judge
		step3_judge_expname = f"{iteration_expname}_step3_judge"
		step3_judge_cmd = (
			"python recipes/robustness-attacks-v3/scripts/step3_judge.py "
			f"--output-folder {output_folder} "
			f"--iteration-id {current_iteration}"
		)
		if judge_model:
			step3_judge_cmd += (
				f" --judge-model {judge_model}"
				f" --judge-endpoint http://localhost:{_MUTATION_SERVER_PORT}/v1"
			)

		step3_judge_kwargs = dict(
			ctx=wrap_arguments(step3_judge_cmd),
			cluster=cluster,
			container=container,
			expname=step3_judge_expname,
			log_dir=f"{log_dir}/iter{current_iteration}/step3_judge",
			run_after=[step1_expname],
		)
		if judge_model:
			step3_judge_kwargs["server_type"] = "vllm"
			step3_judge_kwargs["server_gpus"] = judge_server_gpus
			step3_judge_kwargs["server_nodes"] = judge_server_nodes
			step3_judge_kwargs["server_args"] = f"--max-model-len {judge_max_model_len}"
			step3_judge_kwargs["model"] = judge_model
			step3_judge_kwargs["partition"] = cluster_config.get("partition")
		else:
			step3_judge_kwargs["partition"] = cluster_config.get("cpu_partition")
		run_cmd(**step3_judge_kwargs)

		eval_expnames = [step3_judge_expname]

	return eval_expnames


def run_iterative_attack(
	*,
	cluster: str,
	output_folder: str,
	model_name: str,
	iter_num: int,
	iter_batch_size: int,
	expname_prefix: str,
	log_dir: str,
	container: str,
	benchmark: str,
	subset: str,
	server_gpus: int = 1,
	max_model_len: int = 40960,
	mutation_model: Optional[str] = None,
	mutation_server_gpus: int = 8,
	temperature: float = 1.0,
	seed_base: int = 42,
	eval_mode: str = "benchmark",
	judge_model: Optional[str] = None,
	judge_server_gpus: int = 8,
	judge_server_nodes: int = 1,
	judge_max_model_len: int = 16384,
) -> None:
	"""Submit iterations in batches of iter_batch_size, waiting between batches."""
	setup_logging(disable_hydra_logs=False, use_rich=True)
	cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir=None)

	LOG.info(f"Starting iterative attack: {iter_num} iterations total, batch size {iter_batch_size}")

	prev_eval_expnames: Optional[list] = None
	prev_iteration: Optional[int] = None

	for batch_start in range(1, iter_num + 1, iter_batch_size):
		batch_end = min(batch_start + iter_batch_size - 1, iter_num)
		LOG.info(f"Submitting batch: iterations {batch_start}–{batch_end}")

		last_eval_expnames = None
		last_iteration = None
		for current_iteration in range(batch_start, batch_end + 1):
			if _iteration_complete(cluster_config, output_folder, current_iteration):
				LOG.info(f"Iteration {current_iteration} already complete — skipping")
				prev_eval_expnames = None
				prev_iteration = None
				continue

			eval_expnames = schedule_iteration(
				cluster=cluster,
				cluster_config=cluster_config,
				output_folder=output_folder,
				model_name=model_name,
				current_iteration=current_iteration,
				prev_iteration=prev_iteration,
				expname_prefix=expname_prefix,
				log_dir=log_dir,
				container=container,
				benchmark=benchmark,
				subset=subset,
				server_gpus=server_gpus,
				max_model_len=max_model_len,
				mutation_model=mutation_model,
				mutation_server_gpus=mutation_server_gpus,
				prev_eval_expnames=prev_eval_expnames,
				temperature=temperature,
				seed_base=seed_base,
				eval_mode=eval_mode,
				judge_model=judge_model,
				judge_server_gpus=judge_server_gpus,
				judge_server_nodes=judge_server_nodes,
				judge_max_model_len=judge_max_model_len,
			)
			prev_eval_expnames = eval_expnames
			prev_iteration = current_iteration
			last_eval_expnames = eval_expnames
			last_iteration = current_iteration

		# Trailing step4 for the last iteration in this batch — its step4 was never
		# run inline by a subsequent step1, so we schedule it as a separate job.
		if last_eval_expnames is not None and last_iteration is not None:
			step4_expname = f"{expname_prefix}_iter{last_iteration}_step4"
			step4_cmd = (
				"python recipes/robustness-attacks-v3/scripts/step4.py "
				f"--output-folder {output_folder} "
				f"--iteration-id {last_iteration} "
				f"--eval-mode {eval_mode} "
			)
			last_exp = run_cmd(
				ctx=wrap_arguments(step4_cmd),
				cluster=cluster,
				container=container,
				expname=step4_expname,
				log_dir=f"{log_dir}/iter{last_iteration}/step4",
				run_after=last_eval_expnames,
				partition=cluster_config.get("cpu_partition"),
			)
			LOG.info(f"Waiting for batch {batch_start}–{batch_end} to complete...")
			last_exp._wait_for_jobs(last_exp.jobs)
			LOG.info(f"Batch {batch_start}–{batch_end} completed")
			# Reset for next batch — step4 already ran, no need to re-run
			prev_eval_expnames = None
			prev_iteration = None
		else:
			LOG.info(f"Batch {batch_start}–{batch_end} — all iterations already complete")

	LOG.info(f"All {iter_num} iterations completed successfully!")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run robustness-attack iterations sequentially"
	)
	parser.add_argument(
		"--cluster",
		type=str,
		default="local",
		help="Cluster name to run the jobs on.",
	)
	parser.add_argument(
		"--output-folder",
		type=str,
		required=True,
		help="Folder containing output.jsonl for step1.py.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		required=True,
		help="Model name to be attacked.",
	)
	parser.add_argument(
		"--iter-num",
		type=int,
		required=True,
		help="Total number of iterations to run.",
	)
	parser.add_argument(
		"--iter-batch-size",
		type=int,
		default=1,
		help="Number of iterations to submit before waiting for completion.",
	)
	parser.add_argument(
		"--expname-prefix",
		type=str,
		default="robustness-attack",
		help="Prefix for experiment names of all steps.",
	)
	parser.add_argument(
		"--container",
		type=str,
		default="nemo-skills",
		help="Container to use for the jobs.",
	)
	parser.add_argument(
		"--benchmark",
		type=str,
		default="gpqa",
		help="Benchmark name to use.",
	)
	parser.add_argument(
		"--subset",
		type=str,
		default="diamond",
		help="Benchmark subset name.",
	)
	parser.add_argument(
		"--server-gpus",
		type=int,
		default=1,
		help="Number of GPUs for the vLLM inference server (step3 eval). "
		     "Qwen3-8B fits on 1 GPU; use more only for larger models.",
	)
	parser.add_argument(
		"--max-model-len",
		type=int,
		default=40960,
		help="Max token context length passed to the vLLM server (--max-model-len). "
		     "Must exceed tokens_to_generate (32768) plus prompt tokens.",
	)
	parser.add_argument(
		"--mutation-model",
		type=str,
		default=None,
		help="Path to a local model for distractor mutations (e.g. /hf_models/gpt-oss-120b). "
		     "When set, NeMo-Run starts a vLLM co-process on the step1 GPU node instead of "
		     "calling Azure OpenAI.",
	)
	parser.add_argument(
		"--mutation-server-gpus",
		type=int,
		default=8,
		help="Number of GPUs for the mutation vLLM server. Only used when --mutation-model is set.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="LLM sampling temperature for mutations. Higher values produce more varied outputs.",
	)
	parser.add_argument(
		"--seed-base",
		type=int,
		default=42,
		help="Base value for the iteration seed. Actual seed = seed-base + iteration-id.",
	)
	parser.add_argument(
		"--eval-mode",
		type=str,
		choices=["benchmark", "llm-judge"],
		default="benchmark",
		help="Evaluation mode: 'benchmark' uses ns eval on a benchmark dataset; "
		     "'llm-judge' uses an LLM to directly compare old vs. new distractor.",
	)
	parser.add_argument(
		"--judge-model",
		type=str,
		default=None,
		help="Path to a local model for LLM-as-judge evaluation "
		     "(e.g. /hf_models/llama-70b). When set, NeMo-Run starts a vLLM co-process "
		     "on the step3_judge GPU node instead of calling Azure OpenAI. "
		     "Only used when --eval-mode llm-judge.",
	)
	parser.add_argument(
		"--judge-server-gpus",
		type=int,
		default=8,
		help="Number of GPUs for the judge vLLM server. "
		     "Only used when --eval-mode llm-judge and --judge-model is set.",
	)
	parser.add_argument(
		"--judge-server-nodes",
		type=int,
		default=1,
		help="Number of nodes for the judge vLLM server. "
		     "Set to 2 when using 16+ GPUs (8 GPUs per node). "
		     "Only used when --eval-mode llm-judge and --judge-model is set.",
	)
	parser.add_argument(
		"--judge-max-model-len",
		type=int,
		default=16384,
		help="Max token context length for the judge vLLM server (--max-model-len). "
		     "Only used when --eval-mode llm-judge and --judge-model is set.",
	)

	args = parser.parse_args()

	run_iterative_attack(
		cluster=args.cluster,
		output_folder=args.output_folder,
		model_name=args.model_name,
		iter_num=args.iter_num,
		iter_batch_size=args.iter_batch_size,
		expname_prefix=args.expname_prefix,
		log_dir=f"{args.output_folder}/logs",
		container=args.container,
		benchmark=args.benchmark,
		subset=args.subset,
		server_gpus=args.server_gpus,
		max_model_len=args.max_model_len,
		mutation_model=args.mutation_model,
		mutation_server_gpus=args.mutation_server_gpus,
		temperature=args.temperature,
		seed_base=args.seed_base,
		eval_mode=args.eval_mode,
		judge_model=args.judge_model,
		judge_server_gpus=args.judge_server_gpus,
		judge_server_nodes=args.judge_server_nodes,
		judge_max_model_len=args.judge_max_model_len,
	)


if __name__ == "__main__":
	main()
