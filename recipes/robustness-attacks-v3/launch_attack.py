import argparse
import logging
from typing import Any, Optional

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.utils import get_logger_name, setup_logging


LOG = logging.getLogger(get_logger_name(__file__))


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
	expname_prefix: str,
	log_dir: str,
	container: str,
	benchmark: str,
	subset: str,
	server_gpus: int = 8,
	max_model_len: int = 40960,
	mutation_model: Optional[str] = None,
	mutation_server_gpus: int = 8,
	prev_step4_expname: Optional[str] = None,
	temperature: float = 1.0,
	seed_base: int = 42,
	eval_mode: str = "benchmark",
	judge_model: Optional[str] = None,
	judge_server_gpus: int = 8,
	judge_server_nodes: int = 1,
	judge_max_model_len: int = 16384,
) -> Any:
	"""Schedule all steps for a single iteration. Returns the step4 experiment."""
	iteration_expname = f"{expname_prefix}_iter{current_iteration}"
	step1_expname = f"{iteration_expname}_step1"
	LOG.info(f"Scheduling iteration {current_iteration}")

	# step 1 — wait for previous iteration's step4 if provided
	step1_cmd = (
		"python recipes/robustness-attacks-v3/scripts/step1.py "
		f"--output-folder {output_folder} "
		f"--iteration-id {current_iteration} "
		f"--temperature {temperature} "
		f"--seed-base {seed_base}"
	)
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
		run_after=[prev_step4_expname] if prev_step4_expname else [],
	)
	if mutation_model:
		# Run on a GPU node; NeMo-Run starts a vLLM co-process on the same node.
		step1_kwargs["server_type"] = "vllm"
		step1_kwargs["server_gpus"] = mutation_server_gpus
		step1_kwargs["model"] = mutation_model
		step1_kwargs["partition"] = cluster_config.get("partition")
	else:
		step1_kwargs["partition"] = cluster_config.get("cpu_partition")
	run_cmd(**step1_kwargs)

	if eval_mode == "benchmark":
		# Consolidated step 2 + step 3: single job with shared vLLM server.
		# step3_eval.py runs step2 inline, then launches 15 concurrent evals
		# against one vLLM co-process server (instead of 15 separate jobs).
		step3_expname = f"{iteration_expname}_step3"
		step3_cmd = (
			"python recipes/robustness-attacks-v3/scripts/step3_eval.py "
			f"--output-folder {output_folder} "
			f"--iteration-id {current_iteration} "
			f"--benchmark {benchmark} "
			f"--subset {subset} "
			f"--model {model_name} "
		)
		run_cmd(
			ctx=wrap_arguments(step3_cmd),
			cluster=cluster,
			container=container,
			expname=step3_expname,
			log_dir=f"{log_dir}/iter{current_iteration}/step3",
			run_after=[step1_expname],
			server_type="vllm",
			server_gpus=server_gpus,
			model=model_name,
			server_args=f"--max-model-len {max_model_len}",
			partition=cluster_config.get("partition"),
		)
		step4_run_after = [step3_expname]

	else:  # llm-judge — skip step2 and ns eval; run step3_judge instead
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

		step4_run_after = [step3_judge_expname]

	# step 4 — parse scores and select distractors, runs after all evals complete
	step4_expname = f"{iteration_expname}_step4"
	step4_cmd = (
		"python recipes/robustness-attacks-v3/scripts/step4.py "
		f"--output-folder {output_folder} "
		f"--iteration-id {current_iteration} "
		f"--eval-mode {eval_mode} "
	)
	exp = run_cmd(
		ctx=wrap_arguments(step4_cmd),
		cluster=cluster,
		container=container,
		expname=step4_expname,
		log_dir=f"{log_dir}/iter{current_iteration}/step4",
		run_after=step4_run_after,
		partition=cluster_config.get("cpu_partition"),
	)

	return exp


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
	server_gpus: int = 8,
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

	prev_step4_expname: Optional[str] = None

	for batch_start in range(1, iter_num + 1, iter_batch_size):
		batch_end = min(batch_start + iter_batch_size - 1, iter_num)
		LOG.info(f"Submitting batch: iterations {batch_start}–{batch_end}")

		last_exp = None
		for current_iteration in range(batch_start, batch_end + 1):
			if _iteration_complete(cluster_config, output_folder, current_iteration):
				LOG.info(f"Iteration {current_iteration} already complete — skipping")
				prev_step4_expname = None  # no active job to wait for
				continue

			last_exp = schedule_iteration(
				cluster=cluster,
				cluster_config=cluster_config,
				output_folder=output_folder,
				model_name=model_name,
				current_iteration=current_iteration,
				expname_prefix=expname_prefix,
				log_dir=log_dir,
				container=container,
				benchmark=benchmark,
				subset=subset,
				server_gpus=server_gpus,
				max_model_len=max_model_len,
				mutation_model=mutation_model,
				mutation_server_gpus=mutation_server_gpus,
				prev_step4_expname=prev_step4_expname,
				temperature=temperature,
				seed_base=seed_base,
				eval_mode=eval_mode,
				judge_model=judge_model,
				judge_server_gpus=judge_server_gpus,
				judge_server_nodes=judge_server_nodes,
				judge_max_model_len=judge_max_model_len,
			)
			prev_step4_expname = f"{expname_prefix}_iter{current_iteration}_step4"

		if last_exp is not None:
			LOG.info(f"Waiting for batch {batch_start}–{batch_end} to complete...")
			last_exp._wait_for_jobs(last_exp.jobs)
			LOG.info(f"Batch {batch_start}–{batch_end} completed")
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
		default=8,
		help="Number of GPUs for the vLLM inference server (step3 eval).",
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
