import argparse
import json
import logging
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


def load_score(output_folder: Path, iteration_id: str, distractor_type: str, position: str) -> float:
    """Read symbolic_correct from the metrics.json for a given (type, position) eval."""
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
    return metrics[benchmark_key]["pass@1"]["symbolic_correct"]


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
    """Return candidate if it worsened accuracy (lower is harder), otherwise revert to parent."""
    candidate_score = candidate["eval_score"]
    parent_score = candidate.get("parent_score")

    if parent_score is None:
        # First iteration — no comparison possible, always keep candidate.
        logging.info(
            "  type=%s pos=%s score=%.2f — first iteration, keeping candidate.",
            candidate["type"],
            candidate["position"],
            candidate_score,
        )
        return candidate

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
