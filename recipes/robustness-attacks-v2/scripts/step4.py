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
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["benchmark", "llm-judge"],
        default="benchmark",
        help="Evaluation mode: 'benchmark' reads metrics.json; 'llm-judge' reads result.json.",
    )
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


def load_judge_result(
    output_folder: Path, iteration_id: str, distractor_type: str, position: str
) -> bool:
    """Read judge_more_adversarial from result.json for a given (type, position) slot."""
    result_path = (
        output_folder
        / "logs"
        / f"iter{iteration_id}"
        / "step3_judge"
        / f"type-{distractor_type}_pos-{position}"
        / "result.json"
    )
    if not result_path.exists():
        raise FileNotFoundError(f"Judge result not found: {result_path}")
    with open(result_path) as f:
        return json.load(f)["judge_more_adversarial"]


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


def select_distractor_judge(
    candidate: Dict,
    parent: Optional[Dict],
) -> Dict:
    """Judge-mode selection: keep if judge said MORE adversarial, otherwise revert to parent."""
    if candidate.get("parent_score") is None:
        # First iteration — judge always kept it (see step3_judge).
        logging.info(
            "  type=%s pos=%s — first iteration, keeping candidate.",
            candidate["type"],
            candidate["position"],
        )
        return candidate

    if candidate["eval_score"] == 0.0:
        logging.info(
            "  type=%s pos=%s — judge said MORE adversarial, keeping candidate.",
            candidate["type"],
            candidate["position"],
        )
        return candidate

    # Judge said NOT more adversarial — revert to parent.
    if parent is None:
        logging.warning(
            "  type=%s pos=%s — judge said NOT more adversarial but parent not found; keeping candidate.",
            candidate["type"],
            candidate["position"],
        )
        return candidate

    reverted = dict(parent)
    reverted["eval_score"] = candidate.get("parent_score", 1.0)
    logging.info(
        "  type=%s pos=%s — judge said NOT more adversarial, reverted to parent.",
        candidate["type"],
        candidate["position"],
    )
    return reverted


def main() -> None:
    args = parse_args()
    output_folder = Path(args.output_folder)
    iteration_id = args.iteration_id
    eval_mode = args.eval_mode

    archive_path = output_folder / "archive" / f"iter{iteration_id}.jsonl"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    archive = read_jsonl(archive_path)

    # Derive positions from archive entries (works for both modes).
    _positions = sorted(set(str(item["position"]) for item in archive))

    from constants import DISTRACTOR_TYPES

    # Pre-flight: validate all score sources exist before touching anything.
    missing = []
    if eval_mode == "benchmark":
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
    else:  # llm-judge
        for distractor_type in DISTRACTOR_TYPES:
            for position in _positions:
                result_path = (
                    output_folder / "logs" / f"iter{iteration_id}" / "step3_judge"
                    / f"type-{distractor_type}_pos-{position}" / "result.json"
                )
                if not result_path.exists():
                    missing.append(f"{distractor_type}_pos-{position}")
        if missing:
            raise RuntimeError(
                f"Iteration {iteration_id}: result.json missing for {len(missing)} slot(s) — "
                f"aborting to prevent silent revert-to-parent.\nMissing: {missing}"
            )

    parent_index = build_parent_index(output_folder)

    scored_archive = []
    for item in archive:
        distractor_type = str(item["type"])
        position = str(item["position"])

        if eval_mode == "benchmark":
            score = load_score(output_folder, iteration_id, distractor_type, position)
        else:  # llm-judge: 0.0 = keep (more adversarial), 1.0 = revert
            more_adversarial = load_judge_result(output_folder, iteration_id, distractor_type, position)
            score = 0.0 if more_adversarial else 1.0

        scored_item = dict(item)
        scored_item["eval_score"] = score
        scored_archive.append(scored_item)

    # Overwrite archive with scored entries so scores are preserved in history.
    write_jsonl(archive_path, scored_archive)
    logging.info("Wrote eval_score into archive: %s", archive_path)

    # Selection: dispatch to the appropriate selector for the mode.
    selector = select_distractor_judge if eval_mode == "llm-judge" else select_distractor
    selected = []
    for item in scored_archive:
        distractor_type = str(item["type"])
        position = str(item["position"])
        parent = parent_index.get((distractor_type, position))
        selected.append(selector(item, parent))

    # Write selected distractors as the state for the next iteration.
    current_state_path = output_folder / "current_state.jsonl"
    write_jsonl(current_state_path, selected)
    logging.info("Wrote %d distractors to current_state: %s", len(selected), current_state_path)


if __name__ == "__main__":
    main()
