import argparse
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

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

def run_eval_pipeline(prompt_template: str, dataset_file: str, output_dir: str, model_name: str, cluster: str, expname: str = "reasoning_robustness_eval"):
    pass

def __main__():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline with noise robustness testing')
    parser.add_argument('--model-name', type=str, required=True, help='Model name to use for evaluation')
    parser.add_argument('--cluster', type=str, default='default', help='Cluster name for evaluation')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to store logs')  
    
    args = parser.parse_args()
    
    for noise in NoiseType:
        input_file = 'nemo_skills/dataset/mmlu-redux/test.jsonl'
        output_file = f'nemo_skills/dataset/mmlu-redux/test_with_{noise.name.lower()}.jsonl'
        
        populate_dataset_with_noise(args.cluster, input_file, output_file, noise, log_dir=args.log_dir, expname="reasoning_robustness_eval")
        
        # for prompt in [1, 2, 3]:
            
            # run_eval_pipeline(
            #     prompt_template=f'recipes/reasoning-robustness/prompts/eval-prompt-{prompt}.yaml',
            #     dataset_file=output_file,
            #     model_name=args.model_name
            # )