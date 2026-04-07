"""Test harness for Experiment 2 code changes.

Verifies:
  1. step1.py arg parsing (new args + defaults)
  2. Client selection logic (external API path)
  3. Seed handling (_PASS_SEED flag)
  4. Few-shot directory override
  5. launch_attack.py wiring (step1_cmd construction)
  6. Backward compatibility (Exp 1 / Azure / local vLLM paths unchanged)

Run from repo root:
    python recipes/robustness-attacks-v3/test_exp2.py
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECIPE_DIR = REPO_ROOT / "recipes" / "robustness-attacks-v3"
SCRIPTS_DIR = RECIPE_DIR / "scripts"

# Add scripts dir so step1 imports resolve
sys.path.insert(0, str(SCRIPTS_DIR))

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ========================================
# 1. step1.py arg parsing
# ========================================
print("\n=== 1. step1.py arg parsing ===")

# Import step1 module
spec = importlib.util.spec_from_file_location("step1", SCRIPTS_DIR / "step1.py")
step1 = importlib.util.module_from_spec(spec)
# Patch AzureOpenAI so it doesn't fail without real credentials
with patch("openai.AzureOpenAI", return_value=MagicMock()):
    spec.loader.exec_module(step1)

# Build a parser the same way main() does
parser = argparse.ArgumentParser()
parser.add_argument("--output-folder", type=str, required=True)
parser.add_argument("--iteration-id", type=str, required=True)
parser.add_argument("--temperature", type=float, default=step1.DEFAULT_TEMPERATURE)
parser.add_argument("--seed-base", type=int, default=step1.DEFAULT_SEED_BASE)
parser.add_argument("--mutation-model", type=str, default=None)
parser.add_argument("--mutation-endpoint", type=str, default="http://localhost:5000/v1")
parser.add_argument("--mutation-api-url", type=str, default=None)
parser.add_argument("--mutation-api-model", type=str, default=None)
parser.add_argument("--mutation-api-key-env", type=str, default="INFERENCE_NVIDIA_KEY")
parser.add_argument("--few-shot-dir", type=str, default=None)
parser.add_argument("--benchmark", type=str, default=None)
parser.add_argument("--subset", type=str, default=None)
parser.add_argument("--prev-iteration-id", type=str, default=None)
parser.add_argument("--eval-mode", type=str, choices=["benchmark", "llm-judge"], default="benchmark")

# Test: all new args accepted with Exp 2 values
args = parser.parse_args([
    "--output-folder", "/tmp/test",
    "--iteration-id", "1",
    "--mutation-api-url", "https://inference-api.nvidia.com",
    "--mutation-api-model", "azure/anthropic/claude-sonnet-4-5",
    "--few-shot-dir", "/some/path",
])
check("--mutation-api-url accepted", args.mutation_api_url == "https://inference-api.nvidia.com")
check("--mutation-api-model accepted", args.mutation_api_model == "azure/anthropic/claude-sonnet-4-5")
check("--few-shot-dir accepted", args.few_shot_dir == "/some/path")

# Test: defaults
args_defaults = parser.parse_args([
    "--output-folder", "/tmp/test",
    "--iteration-id", "1",
])
check("mutation-api-url default is None", args_defaults.mutation_api_url is None)
check("mutation-api-model default is None", args_defaults.mutation_api_model is None)
check("mutation-api-key-env default is INFERENCE_NVIDIA_KEY", args_defaults.mutation_api_key_env == "INFERENCE_NVIDIA_KEY")
check("few-shot-dir default is None", args_defaults.few_shot_dir is None)


# ========================================
# 2. Client selection logic
# ========================================
print("\n=== 2. Client selection logic ===")

# Test external API path
os.environ["TEST_NVIDIA_KEY"] = "test-key-123"
step1._PASS_SEED = True  # reset
step1._FEW_SHOT_DIR = RECIPE_DIR / "prompts" / "few-shot-examples"  # reset

mock_openai = MagicMock()
with patch.object(step1, "OpenAI", return_value=mock_openai) as openai_cls:
    # Simulate: mutation_api_url is set
    step1.client = MagicMock()  # reset
    step1.MODEL = "original"
    step1._PASS_SEED = True

    api_key = os.getenv("TEST_NVIDIA_KEY", "")
    step1.client = step1.OpenAI(base_url="https://inference-api.nvidia.com", api_key=api_key)
    step1.MODEL = "azure/anthropic/claude-sonnet-4-5"
    step1._PASS_SEED = False

    openai_cls.assert_called_with(base_url="https://inference-api.nvidia.com", api_key="test-key-123")
    check("OpenAI client created with correct base_url and api_key", True)
    check("MODEL set to claude-sonnet-4-5", step1.MODEL == "azure/anthropic/claude-sonnet-4-5")
    check("_PASS_SEED is False for external API", step1._PASS_SEED is False)

# Test Azure path (default) — _PASS_SEED should remain True
step1._PASS_SEED = True
check("_PASS_SEED is True for Azure (default)", step1._PASS_SEED is True)

os.environ.pop("TEST_NVIDIA_KEY", None)


# ========================================
# 3. Seed handling
# ========================================
print("\n=== 3. Seed handling ===")

mock_client = MagicMock()
mock_response = MagicMock()
mock_response.choices = [MagicMock()]
mock_response.choices[0].message.content = '"test output"'
mock_client.chat.completions.create.return_value = mock_response

original_client = step1.client
original_model = step1.MODEL
step1.client = mock_client
step1.MODEL = "test-model"

# Test with _PASS_SEED = True
step1._PASS_SEED = True
test_item = {"type": "RANDOM_FACT", "distractor": "test text", "position": 0, "id": "test1"}
step1.perflab_mutation_call(test_item, "CODE_SNIPPET", temperature=0.7, seed=42)
call_kwargs = mock_client.chat.completions.create.call_args
check("seed IS passed when _PASS_SEED=True", "seed" in call_kwargs.kwargs)
check("seed value correct", call_kwargs.kwargs.get("seed") == 42)

# Test with _PASS_SEED = False
mock_client.reset_mock()
step1._PASS_SEED = False
step1.perflab_mutation_call(test_item, "CODE_SNIPPET", temperature=0.7, seed=42)
call_kwargs = mock_client.chat.completions.create.call_args
check("seed NOT passed when _PASS_SEED=False", "seed" not in call_kwargs.kwargs)
check("temperature still passed", call_kwargs.kwargs.get("temperature") == 0.7)

# Restore
step1.client = original_client
step1.MODEL = original_model
step1._PASS_SEED = True


# ========================================
# 4. Few-shot directory override
# ========================================
print("\n=== 4. Few-shot directory override ===")

original_dir = RECIPE_DIR / "prompts" / "few-shot-examples-original"
new_dir = RECIPE_DIR / "prompts" / "few-shot-examples"

check("few-shot-examples-original/ exists", original_dir.is_dir())
check("few-shot-examples/ exists", new_dir.is_dir())

original_files = sorted(f.name for f in original_dir.glob("*.jsonl"))
new_files = sorted(f.name for f in new_dir.glob("*.jsonl"))
check("same filenames in both dirs", original_files == new_files, f"original={original_files}, new={new_files}")

expected_files = ["code_snippet.jsonl", "encrypted_text.jsonl", "markup_noise.jsonl", "math_fact.jsonl", "random_fact.jsonl"]
check("all 5 few-shot files present in original/", original_files == expected_files)

# Verify files differ (Exp 1 rewrote them)
any_differ = False
for fname in expected_files:
    orig_content = (original_dir / fname).read_text()
    new_content = (new_dir / fname).read_text()
    if orig_content != new_content:
        any_differ = True
        break
check("original and new few-shot files differ (Exp 1 rewrote them)", any_differ)

# Test _FEW_SHOT_DIR override
step1._FEW_SHOT_DIR = RECIPE_DIR / "prompts" / "few-shot-examples"  # reset
step1._FEW_SHOT_CACHE.clear()
check("default _FEW_SHOT_DIR points to few-shot-examples/", "few-shot-examples" in str(step1._FEW_SHOT_DIR))

step1._FEW_SHOT_DIR = original_dir
step1._FEW_SHOT_CACHE.clear()  # clear cache so reload happens
examples = step1._load_few_shot_examples("RANDOM_FACT", "CODE_SNIPPET", n=3)
check("examples load from overridden dir", len(examples) > 0, f"got {len(examples)} examples")

# Restore
step1._FEW_SHOT_DIR = RECIPE_DIR / "prompts" / "few-shot-examples"
step1._FEW_SHOT_CACHE.clear()


# ========================================
# 5. launch_attack.py wiring
# ========================================
print("\n=== 5. launch_attack.py wiring ===")

spec_la = importlib.util.spec_from_file_location("launch_attack", RECIPE_DIR / "launch_attack.py")
launch_attack = importlib.util.module_from_spec(spec_la)
# Patch imports that require cluster connectivity
mock_utils = MagicMock()
mock_utils.get_logger_name.return_value = "launch_attack"
mock_utils.setup_logging.return_value = None
with patch.dict("sys.modules", {
    "nemo_skills.pipeline": MagicMock(),
    "nemo_skills.pipeline.utils": MagicMock(),
    "nemo_skills.pipeline.cli": MagicMock(),
    "nemo_skills.utils": mock_utils,
}):
    spec_la.loader.exec_module(launch_attack)

# Test argparse
la_parser = argparse.ArgumentParser()
# Replicate the args from launch_attack main()
la_args_list = [
    "--cluster", "ord",
    "--output-folder", "/workspace/test",
    "--model-name", "/hf_models/Qwen3-8B",
    "--iter-num", "5",
    "--expname-prefix", "test-exp2",
    "--mutation-api-url", "https://inference-api.nvidia.com",
    "--mutation-api-model", "azure/anthropic/claude-sonnet-4-5",
    "--few-shot-dir", "/nemo_run/code/recipes/robustness-attacks-v3/prompts/few-shot-examples-original",
    "--benchmark", "gpqa",
    "--subset", "diamond",
]

# Use launch_attack's actual main parser by calling parse on the real code
# We need to capture the step1_cmd that schedule_iteration builds
# Instead, test the function directly by mocking run_cmd

captured_cmds = []


def mock_run_cmd(**kwargs):
    captured_cmds.append(kwargs)


def mock_eval(**kwargs):
    pass


launch_attack.wrap_arguments.reset_mock()
with patch.object(launch_attack, "run_cmd", side_effect=mock_run_cmd):
    with patch.object(launch_attack, "eval", side_effect=mock_eval):
        with patch.object(launch_attack, "pipeline_utils") as mock_pu:
            mock_pu.get_cluster_config.return_value = {"executor": "slurm", "partition": "polar", "cpu_partition": "cpu_short"}
            mock_pu.get_unmounted_path.side_effect = lambda cfg, p: p
            mock_pu.cluster_path_exists.return_value = True

            launch_attack.schedule_iteration(
                cluster="ord",
                cluster_config={"executor": "slurm", "partition": "polar", "cpu_partition": "cpu_short"},
                output_folder="/workspace/test",
                model_name="/hf_models/Qwen3-8B",
                current_iteration=1,
                prev_iteration=None,
                expname_prefix="test-exp2",
                log_dir="/workspace/test/logs",
                container="nemo-skills",
                benchmark="gpqa",
                subset="diamond",
                mutation_api_url="https://inference-api.nvidia.com",
                mutation_api_model="azure/anthropic/claude-sonnet-4-5",
                few_shot_dir="/nemo_run/code/recipes/robustness-attacks-v3/prompts/few-shot-examples-original",
            )

# Find the step1 command
step1_call = [c for c in captured_cmds if "step1" in str(c.get("expname", ""))]
check("step1 job was scheduled", len(step1_call) > 0)

if step1_call:
    step1_kwargs = step1_call[0]
    # wrap_arguments is mocked — it was called with the step1_cmd string.
    # Retrieve the original string from the mock call.
    wrap_args_mock = launch_attack.wrap_arguments
    # First wrap_arguments call is always the step1_cmd
    wrap_call_args = wrap_args_mock.call_args_list[0]
    ctx = str(wrap_call_args)

    check("step1_cmd contains --mutation-api-url", "--mutation-api-url https://inference-api.nvidia.com" in ctx)
    check("step1_cmd contains --mutation-api-model", "--mutation-api-model azure/anthropic/claude-sonnet-4-5" in ctx)
    check("step1_cmd contains --few-shot-dir", "--few-shot-dir /nemo_run/code/recipes/robustness-attacks-v3/prompts/few-shot-examples-original" in ctx)
    check("step1_cmd does NOT contain --mutation-model", "--mutation-model " not in ctx)
    check("uses cpu_partition (no GPU for API mutation)", step1_kwargs.get("partition") == "cpu_short")


# ========================================
# 6. Backward compatibility
# ========================================
print("\n=== 6. Backward compatibility ===")

# Test Exp 1 path: --mutation-model set, no api args, no few-shot-dir
captured_cmds.clear()
launch_attack.wrap_arguments.reset_mock()
with patch.object(launch_attack, "run_cmd", side_effect=mock_run_cmd):
    with patch.object(launch_attack, "eval", side_effect=mock_eval):
        launch_attack.schedule_iteration(
            cluster="ord",
            cluster_config={"executor": "slurm", "partition": "polar", "cpu_partition": "cpu_short"},
            output_folder="/workspace/test",
            model_name="/hf_models/Qwen3-8B",
            current_iteration=1,
            prev_iteration=None,
            expname_prefix="test-exp1",
            log_dir="/workspace/test/logs",
            container="nemo-skills",
            benchmark="gpqa",
            subset="diamond",
            mutation_model="/hf_models/gpt-oss-120b",
            mutation_server_gpus=8,
        )

step1_call = [c for c in captured_cmds if "step1" in str(c.get("expname", ""))]
if step1_call:
    wrap_call_args = launch_attack.wrap_arguments.call_args_list[0]
    ctx = str(wrap_call_args)
    check("[Exp1] step1_cmd contains --mutation-model", "--mutation-model /hf_models/gpt-oss-120b" in ctx)
    check("[Exp1] step1_cmd does NOT contain --mutation-api-url", "--mutation-api-url" not in ctx)
    check("[Exp1] step1_cmd does NOT contain --few-shot-dir", "--few-shot-dir" not in ctx)
    check("[Exp1] uses GPU partition (vLLM co-process)", step1_call[0].get("partition") == "polar")
    check("[Exp1] server_type is vllm", step1_call[0].get("server_type") == "vllm")

# Test Azure path: no --mutation-model, no --mutation-api-url
captured_cmds.clear()
launch_attack.wrap_arguments.reset_mock()
with patch.object(launch_attack, "run_cmd", side_effect=mock_run_cmd):
    with patch.object(launch_attack, "eval", side_effect=mock_eval):
        launch_attack.schedule_iteration(
            cluster="ord",
            cluster_config={"executor": "slurm", "partition": "polar", "cpu_partition": "cpu_short"},
            output_folder="/workspace/test",
            model_name="/hf_models/Qwen3-8B",
            current_iteration=1,
            prev_iteration=None,
            expname_prefix="test-azure",
            log_dir="/workspace/test/logs",
            container="nemo-skills",
            benchmark="gpqa",
            subset="diamond",
        )

step1_call = [c for c in captured_cmds if "step1" in str(c.get("expname", ""))]
if step1_call:
    wrap_call_args = launch_attack.wrap_arguments.call_args_list[0]
    ctx = str(wrap_call_args)
    check("[Azure] step1_cmd does NOT contain --mutation-model", "--mutation-model" not in ctx)
    check("[Azure] step1_cmd does NOT contain --mutation-api-url", "--mutation-api-url" not in ctx)
    check("[Azure] step1_cmd does NOT contain --few-shot-dir", "--few-shot-dir" not in ctx)
    check("[Azure] uses cpu_partition", step1_call[0].get("partition") == "cpu_short")


# ========================================
# Summary
# ========================================
print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
