import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from openai import AzureOpenAI, OpenAI

from utils import wait_for_local_server


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================
# -------- CONFIG ----------
# ==========================

AZURE_KEY = os.getenv("AZURE_KEY", "")
MODEL = os.getenv("AZURE_MODEL", "gpt-5-chat-20250807")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://llm-proxy.perflab.nvidia.com")

_LOCAL_SERVER_PORT = 5000
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_POSITION_DESCRIPTIONS = {
    "0": "inserted before the question text",
    "1": "inserted after the question text but before the answer options",
    "2": "inserted after the answer options",
}

# ==========================
# ----- Azure Client -------
# ==========================

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
)

# ==========================
# ---- File Utilities ------
# ==========================

def read_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ==========================
# ----- Judge Logic --------
# ==========================

def _judge_single_call(
    judge_prompt: dict,
    distractor_type: str,
    position_description: str,
    distractor_a: str,
    distractor_b: str,
) -> str:
    """Make a single judge LLM call and return the raw response."""
    user_msg = judge_prompt["user"].format(
        distractor_type=distractor_type,
        position_description=position_description,
        distractor_a=distractor_a,
        distractor_b=distractor_b,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    return response.choices[0].message.content.strip()


def call_judge(
    candidate: Dict,
    parent: Dict,
    judge_prompt: dict,
) -> tuple[bool, str, str]:
    """Call the LLM judge twice with swapped order to counter positional bias.

    Forward call:  A=old, B=new  → "B" means new is more adversarial
    Reverse call:  A=new, B=old  → "A" means new is more adversarial

    Returns (more_adversarial, raw_forward, raw_reverse).
    Accept if *either* ordering says the new distractor is more adversarial.
    """
    pos_desc = _POSITION_DESCRIPTIONS.get(
        str(candidate["position"]), str(candidate["position"])
    )

    # Forward: A=old, B=new — answer "B" means new wins
    raw_forward = _judge_single_call(
        judge_prompt, candidate["type"], pos_desc,
        distractor_a=parent["distractor"],
        distractor_b=candidate["distractor"],
    )
    forward_new_wins = "B" in raw_forward.upper()

    # Reverse: A=new, B=old — answer "A" means new wins
    raw_reverse = _judge_single_call(
        judge_prompt, candidate["type"], pos_desc,
        distractor_a=candidate["distractor"],
        distractor_b=parent["distractor"],
    )
    reverse_new_wins = "A" in raw_reverse.upper()

    more_adversarial = forward_new_wins or reverse_new_wins
    return more_adversarial, raw_forward, raw_reverse


# ==========================
# --------- MAIN -----------
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for robustness-attacks.")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--iteration-id", type=str, required=True)
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Path to local model for judging. When set, uses a co-process vLLM server.",
    )
    parser.add_argument(
        "--judge-endpoint",
        type=str,
        default=f"http://localhost:{_LOCAL_SERVER_PORT}/v1",
        help="OpenAI-compatible endpoint for the local judge server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.judge_model:
        wait_for_local_server(args.judge_endpoint)
        global client, MODEL
        client = OpenAI(base_url=args.judge_endpoint, api_key="EMPTY")
        MODEL = args.judge_model
        logging.info("Using local judge model: %s @ %s", args.judge_model, args.judge_endpoint)
    else:
        logging.info("Using Azure judge model: %s @ %s", MODEL, AZURE_ENDPOINT)

    output_folder = Path(args.output_folder)
    iteration_id = args.iteration_id

    # Load judge prompt from YAML
    judge_prompt = yaml.safe_load(open(_PROMPTS_DIR / "judge-prompt.yaml"))

    # Load candidates (produced by step1)
    archive_path = output_folder / "archive" / f"iter{iteration_id}.jsonl"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    archive = read_jsonl(archive_path)

    # Build parent index from current_state.jsonl (pre-step4 state = parents)
    parent_state_path = output_folder / "current_state.jsonl"
    parent_index: Dict[tuple, Dict] = {}
    if parent_state_path.exists():
        for item in read_jsonl(parent_state_path):
            parent_index[(str(item["type"]), str(item["position"]))] = item

    logging.info(
        "Iteration %s: judging %d candidates against %d parents",
        iteration_id, len(archive), len(parent_index),
    )

    for candidate in archive:
        distractor_type = str(candidate["type"])
        position = str(candidate["position"])
        parent: Optional[Dict] = parent_index.get((distractor_type, position))

        result_path = (
            output_folder / "logs" / f"iter{iteration_id}"
            / "step3_judge"
            / f"type-{distractor_type}_pos-{position}"
            / "result.json"
        )

        if parent is None or candidate.get("parent_score") is None:
            # First iteration or no parent found — always keep candidate.
            logging.info(
                "  type=%s pos=%s — first iteration, keeping candidate without judge call.",
                distractor_type, position,
            )
            result = {
                "type": distractor_type,
                "position": position,
                "distractor_id": candidate["id"],
                "parent_id": candidate.get("parent_id"),
                "judge_more_adversarial": True,
                "judge_raw_response": "FIRST_ITERATION",
                "old_distractor": parent["distractor"] if parent else None,
                "new_distractor": candidate["distractor"],
            }
        else:
            more_adversarial, raw_forward, raw_reverse = call_judge(candidate, parent, judge_prompt)
            logging.info(
                "  type=%s pos=%s — judge forward=%r reverse=%r → more_adversarial=%s",
                distractor_type, position, raw_forward, raw_reverse, more_adversarial,
            )
            result = {
                "type": distractor_type,
                "position": position,
                "distractor_id": candidate["id"],
                "parent_id": candidate.get("parent_id"),
                "judge_more_adversarial": more_adversarial,
                "judge_raw_response_forward": raw_forward,
                "judge_raw_response_reverse": raw_reverse,
                "old_distractor": parent["distractor"],
                "new_distractor": candidate["distractor"],
            }

        write_json(result_path, result)
        logging.info("  Wrote result to %s", result_path)

    logging.info("Step3-judge completed for iteration %s", iteration_id)


if __name__ == "__main__":
    main()
