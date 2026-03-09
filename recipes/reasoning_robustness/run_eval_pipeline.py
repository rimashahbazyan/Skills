import argparse
from nemo_skills.pipeline.cli import eval, run_cmd, wrap_arguments

from recipes.reasoning_robustness.scripts.noise_constants import NoiseType


def populate_dataset_with_noise(cluster: str, input_file: str, output_file: str, noise: NoiseType, log_dir: str, expname: str):
    cmd = (
        f"python /nemo_run/code/recipes/reasoning_robustness/scripts/populate_dataset_with_noises.py "
        f"--input {input_file} "
        f"--output {output_file} "
        f"--noise \"{noise.value}\""
    )

    run_cmd(
        cluster=cluster,
        container="nemo-skills",
        log_dir=f"{log_dir}/populate_dataset/{noise.name.lower()}",
        expname=f"{expname}_populate_{noise.name.lower()}",
        ctx=wrap_arguments(cmd),
    )
    print(f"Populated dataset with noise '{noise.name}' and saved to {output_file}")


def run_eval_pipeline(
    prompt_id: int,
    dataset: str,
    output_dir: str,
    model_name: str,
    cluster: str,
    log_dir: str,
    expname: str = "reasoning_robustness_eval",
):
    prompt_template=f'/nemo_run/code/recipes/reasoning_robustness/prompts/eval-prompt-{prompt_id}.yaml'


    eval(
        ctx=wrap_arguments(
            f"++inference.tokens_to_generate={32000} "
            f"++max_concurrent_requests=512 "
            f"++inference.endpoint_type=text "
            f"++skip_filled=True "
            f"++prompt_config={prompt_template} "

        ),
        cluster=cluster,
        expname=f"{expname}_eval_{dataset}_prompt{prompt_id}",
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        model=model_name,
        server_type="vllm",
        server_gpus=8,
        benchmarks=f"mmlu-redux-long:{1}",
        num_chunks=5,
        # dependent_jobs=1,
        split=dataset,
        extra_datasets="/workspace/Robustness-MMLU-Redux/datasets",
        extra_datasets_type="cluster",
    )


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline with noise robustness testing")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to use for evaluation")
    parser.add_argument("--cluster", type=str, default="default", help="Cluster name for evaluation")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store evaluation outputs (defaults to --log-dir if not provided)",
    )

    args = parser.parse_args()
    base_output_dir = args.output_dir or args.log_dir

    for noise in NoiseType:
        input_file = "nemo_skills/dataset/mmlu-redux/test.jsonl"
        output_file = f"/workspace/Robustness-MMLU-Redux/datasets/mmlu-redux-long/test_with_{noise.name.lower()}.jsonl"

        # populate_dataset_with_noise(
        #     args.cluster,
        #     input_file,
        #     output_file,
        #     noise,
        #     log_dir=f"{base_output_dir}/populate_dataset/{noise.name.lower()}",
        #     expname=f"reasoning_robustness_eval_populate_{noise.name.lower()}",
        # )

        for prompt_id in [1, 2, 3]:
            run_eval_pipeline(
                prompt_id=prompt_id,
                dataset=f"test_with_{noise.name.lower()}",
                output_dir=f"{base_output_dir}/eval_{noise.name.lower()}_prompt{prompt_id}",
                model_name=args.model_name,
                cluster=args.cluster,
                log_dir=f"{base_output_dir}/eval_{noise.name.lower()}_prompt{prompt_id}/logs",
            )

        
        #     break  # Remove this break to run for all noise types
        # break  # Remove this break to run for all noise types

        # default eval

    #         eval(
    #     ctx=wrap_arguments(
    #         f"++inference.tokens_to_generate={32000} "
    #         f"++max_concurrent_requests=512 "
    #         f"++inference.endpoint_type=text "
    #         f"++skip_filled=True "

    #     ),
    #     cluster=args.cluster,
    #     expname=f"reasoning_robustness_eval_gpqa",
    #     output_dir=f"{base_output_dir}/eval_default",
    #     log_dir=f"{base_output_dir}/eval_default/logs",
    #     model=args.model_name,
    #     server_type="vllm",
    #     server_gpus=8,
    #     benchmarks=f"gpqa:{1}",
    #     num_chunks=5,
    #     # dependent_jobs=1,
    #     split="diamond",
    #     extra_datasets="/workspace/Robustness-GPQA/datasets",
    #     extra_datasets_type="cluster",
    # )



if __name__ == "__main__":
    main()
