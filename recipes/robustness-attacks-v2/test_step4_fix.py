"""Test that step4's load_score + select_distractor correctly handle failed evaluations.

Reproduces the real failure from iter67/type-CODE_SNIPPET_pos-2 where the model
produced empty generations (no_answer=100%, symbolic_correct=0.0, num_entries=198).
Without the fix, step4 would accept this as a successful attack (0.0 < parent_score).
With the fix, it should revert to the parent.
"""
import json
import sys
import tempfile
from pathlib import Path

# Make scripts/ importable so step4 can find constants
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.step4 import load_score, select_distractor


def _write_metrics(output_folder, iteration_id, distractor_type, position, metrics_data):
    """Write a metrics.json file at the expected path."""
    metrics_dir = (
        output_folder
        / "logs"
        / f"iter{iteration_id}"
        / "step3"
        / f"type-{distractor_type}_pos-{position}"
        / "eval-results"
        / f"iter{iteration_id}"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f)


def test_failed_eval_no_answer_100():
    """Reproduce iter67 failure: num_entries=198, no_answer=100%, symbolic_correct=0.0"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        _write_metrics(output_folder, "67", "CODE_SNIPPET", "2", {
            "iter67": {
                "pass@1": {
                    "num_entries": 198,
                    "gen_seconds": 6,
                    "symbolic_correct": 0.0,
                    "no_answer": 100.0,
                }
            }
        })

        score = load_score(output_folder, "67", "CODE_SNIPPET", "2")
        assert score is None, f"Expected None for failed eval, got {score}"
        print("PASS: load_score returns None for no_answer=100%")


def test_failed_eval_num_entries_zero():
    """Edge case: num_entries=0 (eval produced nothing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        _write_metrics(output_folder, "1", "RANDOM_FACT", "0", {
            "iter1": {
                "pass@1": {
                    "num_entries": 0,
                    "symbolic_correct": 0.0,
                }
            }
        })

        score = load_score(output_folder, "1", "RANDOM_FACT", "0")
        assert score is None, f"Expected None for num_entries=0, got {score}"
        print("PASS: load_score returns None for num_entries=0")


def test_genuine_zero_accuracy():
    """Real 0% accuracy (all wrong but model did generate answers) should NOT be treated as failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        _write_metrics(output_folder, "1", "RANDOM_FACT", "0", {
            "iter1": {
                "pass@1": {
                    "num_entries": 198,
                    "gen_seconds": 120,
                    "symbolic_correct": 0.0,
                    "no_answer": 5.0,  # some no_answer but not 100%
                }
            }
        })

        score = load_score(output_folder, "1", "RANDOM_FACT", "0")
        assert score == 0.0, f"Expected 0.0 for genuine zero accuracy, got {score}"
        print("PASS: load_score returns 0.0 for genuine zero accuracy (no_answer < 100%)")


def test_normal_score():
    """Normal evaluation with real accuracy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        _write_metrics(output_folder, "1", "RANDOM_FACT", "0", {
            "iter1": {
                "pass@1": {
                    "num_entries": 198,
                    "gen_seconds": 600,
                    "symbolic_correct": 45.5,
                    "no_answer": 2.0,
                }
            }
        })

        score = load_score(output_folder, "1", "RANDOM_FACT", "0")
        assert score == 45.5, f"Expected 45.5, got {score}"
        print("PASS: load_score returns correct score for normal eval")


def test_select_distractor_reverts_on_failed_eval():
    """select_distractor should revert to parent when eval_score is None."""
    candidate = {
        "id": "67_CODE_SNIPPET_2",
        "type": "CODE_SNIPPET",
        "position": "2",
        "distractor": "new distractor text",
        "eval_score": None,  # failed eval
        "parent_score": 35.0,
        "parent_id": "50_CODE_SNIPPET_2",
    }
    parent = {
        "id": "50_CODE_SNIPPET_2",
        "type": "CODE_SNIPPET",
        "position": "2",
        "distractor": "parent distractor text",
        "eval_score": 35.0,
    }

    result = select_distractor(candidate, parent)
    assert result["id"] == parent["id"], f"Expected revert to parent, got {result['id']}"
    assert result["eval_score"] == 35.0
    print("PASS: select_distractor reverts to parent on failed eval (None score)")


def test_select_distractor_accepts_genuine_improvement():
    """select_distractor should accept when score genuinely decreased."""
    candidate = {
        "id": "67_CODE_SNIPPET_2",
        "type": "CODE_SNIPPET",
        "position": "2",
        "distractor": "better attack",
        "eval_score": 20.0,
        "parent_score": 35.0,
        "parent_id": "50_CODE_SNIPPET_2",
    }
    parent = {
        "id": "50_CODE_SNIPPET_2",
        "type": "CODE_SNIPPET",
        "position": "2",
        "distractor": "parent distractor text",
        "eval_score": 35.0,
    }

    result = select_distractor(candidate, parent)
    assert result["id"] == candidate["id"], f"Expected candidate kept, got {result['id']}"
    assert result["eval_score"] == 20.0
    print("PASS: select_distractor keeps candidate with genuinely lower score")


if __name__ == "__main__":
    test_failed_eval_no_answer_100()
    test_failed_eval_num_entries_zero()
    test_genuine_zero_accuracy()
    test_normal_score()
    test_select_distractor_reverts_on_failed_eval()
    test_select_distractor_accepts_genuine_improvement()
    print("\nAll tests passed!")
