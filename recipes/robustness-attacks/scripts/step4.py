import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse eval results and select best distractors.")
    parser.add_argument("--output-folder", type=str, required=True, help="Root output folder.")
    parser.add_argument("--iteration-id", type=str, required=True, help="Current iteration id.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_score(
    output_folder: Path, iteration_id: str, distractor_type: str, position: str
) -> Optional[float]:
    """Read symbolic_correct from the metrics.json for a given (type, position) eval.

    Returns None if the evaluation failed, so that step4 can revert to the
    parent instead of treating 0.0 as a successful attack.  A failure is
    detected when either:
      - num_entries == 0  (no questions evaluated at all), or
      - no_answer == 100  (model produced empty generations for every question,
        e.g. due to server timeouts).
    """
    metrics_path = (
        output_folder
        / "logs"
        / f"iter{iteration_id}"
        / "step3"
        / f"type-{distractor_type}_pos-{position}"
        / "eval-results"
        / f"iter{iteration_id}"
        / "metrics.json"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    benchmark_key = f"iter{iteration_id}"
    pass_at_1 = metrics[benchmark_key]["pass@1"]

    num_entries = pass_at_1.get("num_entries", 0)
    no_answer = pass_at_1.get("no_answer", 0.0)

    if num_entries == 0:
        logging.warning(
            "  type=%s pos=%s — eval has num_entries=0, treating as failed evaluation.",
            distractor_type, position,
        )
        return None

    if no_answer == 100.0:
        logging.warning(
            "  type=%s pos=%s — eval has no_answer=100%% (%d entries, all empty generations), "
            "treating as failed evaluation.",
            distractor_type, position, num_entries,
        )
        return None

    return pass_at_1["symbolic_correct"]


def build_parent_index(output_folder: Path) -> Dict[Tuple[str, str], Dict]:
    """Load current_state.jsonl as the parent index for reversion.

    current_state.jsonl always holds the correct parent state when step4 runs:
    step1 reads it but never modifies it, so it still reflects what was mutated
    this iteration regardless of which iteration number we are on.
    """
    parent_path = output_folder / "current_state.jsonl"
    if not parent_path.exists():
        logging.warning("current_state.jsonl not found — reversion will not be possible.")
        return {}

    parent_state = read_jsonl(parent_path)
    return {(str(item["type"]), str(item["position"])): item for item in parent_state}


def select_distractor(
    candidate: Dict,
    parent: Optional[Dict],
) -> Dict:
    """Return candidate if it worsened accuracy (lower is harder), otherwise revert to parent.

    If eval_score is None (failed evaluation), reverts to parent to avoid
    treating a broken eval as a successful attack.
    """
    candidate_score = candidate["eval_score"]
    parent_score = candidate.get("parent_score")

    if parent_score is None:
        # First iteration — no comparison possible, always keep candidate.
        logging.info(
            "  type=%s pos=%s score=%s — first iteration, keeping candidate.",
            candidate["type"],
            candidate["position"],
            candidate_score,
        )
        return candidate

    # Failed evaluation — revert to parent rather than treating 0.0 as a win.
    if candidate_score is None:
        if parent is None:
            logging.warning(
                "  type=%s pos=%s — eval failed and parent not found; keeping candidate.",
                candidate["type"],
                candidate["position"],
            )
            return candidate
        reverted = dict(parent)
        reverted["eval_score"] = parent_score
        logging.warning(
            "  type=%s pos=%s — eval failed (num_entries=0), reverted to parent (score=%.2f).",
            candidate["type"],
            candidate["position"],
            parent_score,
        )
        return reverted

    if candidate_score < parent_score:
        logging.info(
            "  type=%s pos=%s score=%.2f < parent=%.2f — kept (attack improved).",
            candidate["type"],
            candidate["position"],
            candidate_score,
            parent_score,
        )
        return candidate

    # Mutation did not worsen accuracy — revert to parent.
    if parent is None:
        logging.warning(
            "  type=%s pos=%s score=%.2f >= parent=%.2f — would revert but parent not found; keeping candidate.",
            candidate["type"],
            candidate["position"],
            candidate_score,
            parent_score,
        )
        return candidate

    reverted = dict(parent)
    reverted["eval_score"] = parent_score
    logging.info(
        "  type=%s pos=%s score=%.2f >= parent=%.2f — reverted to parent.",
        candidate["type"],
        candidate["position"],
        candidate_score,
        parent_score,
    )
    return reverted


def main() -> None:
    args = parse_args()
    output_folder = Path(args.output_folder)
    iteration_id = args.iteration_id

    # Derive positions from the eval-prompt-*.yaml files (same logic as launch_attack.py)
    _prompt_pattern = os.path.join(os.path.dirname(__file__), "..", "prompts", "eval-prompt-*.yaml")
    _positions = sorted(
        set(
            os.path.splitext(os.path.basename(p))[0].split("eval-prompt-")[1]
            for p in glob.glob(_prompt_pattern)
        )
    )

    # Validate all metrics.json files exist before touching anything — fail loudly
    # so the Slurm job is marked FAILED instead of silently reverting slots.
    from constants import DISTRACTOR_TYPES
    missing = []
    for distractor_type in DISTRACTOR_TYPES:
        for position in _positions:
            metrics_path = (
                output_folder / "logs" / f"iter{iteration_id}" / "step3"
                / f"type-{distractor_type}_pos-{position}"
                / "eval-results" / f"iter{iteration_id}" / "metrics.json"
            )
            if not metrics_path.exists():
                missing.append(f"{distractor_type}_pos-{position}")
    if missing:
        raise RuntimeError(
            f"Iteration {iteration_id}: metrics.json missing for {len(missing)} slot(s) — "
            f"aborting to prevent silent revert-to-parent.\nMissing: {missing}"
        )

    archive_path = output_folder / "archive" / f"iter{iteration_id}.jsonl"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    archive = read_jsonl(archive_path)
    parent_index = build_parent_index(output_folder)

    scored_archive = []
    for item in archive:
        distractor_type = str(item["type"])
        position = str(item["position"])

        score = load_score(output_folder, iteration_id, distractor_type, position)
        scored_item = dict(item)
        scored_item["eval_score"] = score
        scored_archive.append(scored_item)

    # Overwrite archive with scored entries so scores are preserved in history.
    write_jsonl(archive_path, scored_archive)
    logging.info("Wrote eval_score into archive: %s", archive_path)

    # Selection: keep mutation only if accuracy worsened.
    selected = []
    for item in scored_archive:
        distractor_type = str(item["type"])
        position = str(item["position"])
        parent = parent_index.get((distractor_type, position))
        selected.append(select_distractor(item, parent))

    # Write selected distractors as the state for the next iteration.
    current_state_path = output_folder / "current_state.jsonl"
    write_jsonl(current_state_path, selected)
    logging.info("Wrote %d distractors to current_state: %s", len(selected), current_state_path)


if __name__ == "__main__":
    main()
