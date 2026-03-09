import argparse
import glob
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
) -> Any:
	"""Schedule all steps for a single iteration. Returns the step4 experiment."""
	iteration_expname = f"{expname_prefix}_iter{current_iteration}"
	step1_expname = f"{iteration_expname}_step1"
	step2_expname = f"{iteration_expname}_step2"

	LOG.info(f"Scheduling iteration {current_iteration}")

	# step 1 — wait for previous iteration's step4 if provided
	step1_cmd = (
		"python recipes/robustness-attacks/scripts/step1.py "
		f"--output-folder {output_folder} "
		f"--iteration-id {current_iteration}"
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
		step1_kwargs["partition"] = "cpu_short"
	run_cmd(**step1_kwargs)

	# step 2
	step2_cmd = (
		"python recipes/robustness-attacks/scripts/step2.py "
		f"--iter {current_iteration} "
		f"--benchmark {benchmark} "
		f"--subset {subset} "
		f"--output-folder {output_folder} "
	)
	run_cmd(
		ctx=wrap_arguments(step2_cmd),
		cluster=cluster,
		container=container,
		expname=step2_expname,
		log_dir=f"{log_dir}/iter{current_iteration}/step2",
		run_after=[step1_expname],
		partition="cpu_short",
	)

	# calculate number of positions from prompts
	pattern = "recipes/robustness-attacks/prompts/eval-prompt-*.yaml"
	prompt_files = glob.glob(pattern)
	positions = sorted(
		set(
			os.path.splitext(os.path.basename(file_path))[0].split("eval-prompt-")[1]
			for file_path in prompt_files
		)
	)

	# Ensure iter{N}/__init__.py and stub data files exist before eval() submission checks.
	_upload_iter_stubs(cluster_config, output_folder, current_iteration, benchmark, subset, DISTRACTOR_TYPES, positions)

	eval_expnames = []
	for distractor_type in DISTRACTOR_TYPES:
		for position in positions:
			split = f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}"
			prompt_template = f'/nemo_run/code/recipes/robustness-attacks/prompts/eval-prompt-{position}.yaml'
			eval_expname = f"{iteration_expname}_eval_type-{distractor_type}_pos-{position}"
			eval_expnames.append(eval_expname)

			# step 3
			eval(
				ctx=wrap_arguments(
					f"++inference.tokens_to_generate={32768} "
					f"++max_concurrent_requests=512 "
					f"++inference.endpoint_type=text "
					f"++skip_filled=True "
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
				# dependent_jobs=1,
				split=split,
				data_dir=f"{output_folder}",
				run_after=[step2_expname],
			)

	# step 4 — parse scores and select distractors, runs after all evals complete
	step4_expname = f"{iteration_expname}_step4"
	step4_cmd = (
		"python recipes/robustness-attacks/scripts/step4.py "
		f"--output-folder {output_folder} "
		f"--iteration-id {current_iteration} "
	)
	exp = run_cmd(
		ctx=wrap_arguments(step4_cmd),
		cluster=cluster,
		container=container,
		expname=step4_expname,
		log_dir=f"{log_dir}/iter{current_iteration}/step4",
		run_after=eval_expnames,
		partition="cpu_short",
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
			)
			prev_step4_expname = f"{expname_prefix}_iter{current_iteration}_step4"

		LOG.info(f"Waiting for batch {batch_start}–{batch_end} to complete...")
		last_exp._wait_for_jobs(last_exp.jobs)
		LOG.info(f"Batch {batch_start}–{batch_end} completed")

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
	)


if __name__ == "__main__":
	main()
