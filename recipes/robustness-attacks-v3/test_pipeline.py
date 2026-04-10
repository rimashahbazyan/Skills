"""Pytest unit tests for the robustness-attacks-v3 pipeline.

Covers step1 (mutation), step2 (injection), step4 (selection) logic.
No external dependencies (no GPU, no cluster, no API keys).

Run: pytest recipes/robustness-attacks-v3/test_pipeline.py -s -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add scripts dir to path so we can import step1/step2/step4
_SCRIPTS_DIR = str(Path(__file__).parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from step4 import select_distractor, select_distractor_judge, load_score, build_parent_index, run_summarize_results, read_jsonl, write_jsonl
from step2 import validate_and_index_distractors, write_injected_files
from step1 import _pick_source_for_slot, _format_examples, create_initial_state, load_state, save_state


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_state():
    """Minimal 3-type x 2-position grid (6 slots)."""
    return [
        {"id": "RF_0", "type": "RANDOM_FACT", "position": 0, "distractor": "Honey never spoils."},
        {"id": "RF_1", "type": "RANDOM_FACT", "position": 1, "distractor": "Venus day is longer than year."},
        {"id": "CS_0", "type": "CODE_SNIPPET", "position": 0, "distractor": "def foo(): pass"},
        {"id": "CS_1", "type": "CODE_SNIPPET", "position": 1, "distractor": "x = [1, 2, 3]"},
        {"id": "ET_0", "type": "ENCRYPTED_TEXT", "position": 0, "distractor": "7hN3xQm9K2pL5v"},
        {"id": "ET_1", "type": "ENCRYPTED_TEXT", "position": 1, "distractor": "aG9uZXkgbmV2ZXI="},
    ]


@pytest.fixture
def candidate():
    return {
        "id": "RF_0_new",
        "type": "RANDOM_FACT",
        "position": 0,
        "distractor": "New distractor text",
        "eval_score": 45.0,
        "parent_score": 55.0,
        "parent_id": "RF_0",
    }


@pytest.fixture
def parent():
    return {
        "id": "RF_0",
        "type": "RANDOM_FACT",
        "position": 0,
        "distractor": "Original distractor text",
        "eval_score": 55.0,
    }


# ============================================================
# 1. Pure function tests — select_distractor
# ============================================================

class TestSelectDistractor:
    def test_keeps_when_score_improves(self, candidate, parent):
        """candidate_score < parent_score → attack improved, keep candidate."""
        candidate["eval_score"] = 40.0
        candidate["parent_score"] = 55.0
        result = select_distractor(candidate, parent)
        assert result["id"] == candidate["id"]
        assert result["eval_score"] == 40.0

    def test_reverts_when_score_worsens(self, candidate, parent):
        """candidate_score > parent_score → attack worsened, revert."""
        candidate["eval_score"] = 65.0
        candidate["parent_score"] = 55.0
        result = select_distractor(candidate, parent)
        assert result["id"] == parent["id"]
        assert result["eval_score"] == 55.0

    def test_reverts_when_score_equal(self, candidate, parent):
        """candidate_score == parent_score → no improvement, revert."""
        candidate["eval_score"] = 55.0
        candidate["parent_score"] = 55.0
        result = select_distractor(candidate, parent)
        assert result["id"] == parent["id"]

    def test_reverts_on_failed_eval(self, candidate, parent):
        """eval_score is None (eval failed) → revert to parent."""
        candidate["eval_score"] = None
        candidate["parent_score"] = 55.0
        result = select_distractor(candidate, parent)
        assert result["id"] == parent["id"]
        assert result["eval_score"] == 55.0

    def test_keeps_first_iteration(self, candidate, parent):
        """parent_score is None (first iteration) → always keep."""
        candidate["parent_score"] = None
        candidate["eval_score"] = 70.0
        result = select_distractor(candidate, parent)
        assert result["id"] == candidate["id"]

    def test_keeps_when_parent_missing(self, candidate):
        """parent is None → keep candidate (with warning)."""
        candidate["eval_score"] = 65.0
        candidate["parent_score"] = 55.0
        result = select_distractor(candidate, None)
        assert result["id"] == candidate["id"]


# ============================================================
# 1b. Pure function tests — select_distractor_judge
# ============================================================

class TestSelectDistractorJudge:
    def test_keeps_when_more_adversarial(self, candidate, parent):
        """eval_score == 0.0 → judge said more adversarial, keep."""
        candidate["eval_score"] = 0.0
        candidate["parent_score"] = 1.0
        result = select_distractor_judge(candidate, parent)
        assert result["id"] == candidate["id"]

    def test_reverts_when_not_adversarial(self, candidate, parent):
        """eval_score == 1.0 → judge said not adversarial, revert."""
        candidate["eval_score"] = 1.0
        candidate["parent_score"] = 0.0
        result = select_distractor_judge(candidate, parent)
        assert result["id"] == parent["id"]

    def test_keeps_first_iteration(self, candidate, parent):
        """parent_score is None → first iteration, always keep."""
        candidate["parent_score"] = None
        result = select_distractor_judge(candidate, parent)
        assert result["id"] == candidate["id"]


# ============================================================
# 1c. Pure function tests — _pick_source_for_slot
# ============================================================

class TestPickSourceForSlot:
    def test_excludes_same_type(self, sample_state):
        """Never picks a distractor of the same type."""
        for _ in range(50):  # run many times since random
            picked = _pick_source_for_slot("RANDOM_FACT", sample_state, set())
            assert picked["type"] != "RANDOM_FACT"

    def test_excludes_used_ids(self, sample_state):
        """Respects used_ids — doesn't pick already-used IDs."""
        used = {"CS_0", "CS_1", "ET_0"}  # all non-RANDOM_FACT except ET_1
        for _ in range(50):
            picked = _pick_source_for_slot("RANDOM_FACT", sample_state, used)
            assert picked["id"] == "ET_1"

    def test_falls_back_when_all_used(self, sample_state):
        """Falls back to allowing reuse when all different-type IDs are exhausted."""
        used = {"CS_0", "CS_1", "ET_0", "ET_1"}  # all non-RANDOM_FACT used
        picked = _pick_source_for_slot("RANDOM_FACT", sample_state, used)
        assert picked["type"] != "RANDOM_FACT"  # still cross-type


# ============================================================
# 1d. Pure function tests — _format_examples
# ============================================================

class TestFormatExamples:
    def test_with_examples(self):
        examples = [
            {"initial_distractor": "source text", "mutated_distractor": "target text"},
        ]
        result = _format_examples(examples, "Random Fact", "Code Snippet")
        assert "Random Fact" in result
        assert "Code Snippet" in result
        assert "source text" in result
        assert "target text" in result

    def test_empty_examples(self):
        result = _format_examples([], "Random Fact", "Code Snippet")
        assert "Here are examples" in result
        assert "---" not in result.split("\n", 1)[1]  # no example separators


# ============================================================
# 2. File I/O tests — validate_and_index_distractors
# ============================================================

class TestValidateAndIndexDistractors:
    def _write_distractors(self, path, distractors):
        with open(path, "w") as f:
            for d in distractors:
                f.write(json.dumps(d) + "\n")

    def test_full_grid(self, tmp_path):
        """Valid 2x2 grid passes validation."""
        distractors = [
            {"id": "A_0", "type": "A", "position": 0, "distractor": "a0"},
            {"id": "A_1", "type": "A", "position": 1, "distractor": "a1"},
            {"id": "B_0", "type": "B", "position": 0, "distractor": "b0"},
            {"id": "B_1", "type": "B", "position": 1, "distractor": "b1"},
        ]
        path = tmp_path / "test.jsonl"
        self._write_distractors(path, distractors)
        types, positions, index = validate_and_index_distractors(path)
        assert types == ["A", "B"]
        assert positions == ["0", "1"]
        assert len(index) == 4

    def test_missing_keys(self, tmp_path):
        """Raises ValueError when required keys are missing."""
        distractors = [{"id": "A_0", "type": "A"}]  # missing position, distractor
        path = tmp_path / "test.jsonl"
        self._write_distractors(path, distractors)
        with pytest.raises(ValueError, match="Missing required keys"):
            validate_and_index_distractors(path)

    def test_duplicate_slot(self, tmp_path):
        """Raises ValueError on duplicate (type, position)."""
        distractors = [
            {"id": "A_0a", "type": "A", "position": 0, "distractor": "a0a"},
            {"id": "A_0b", "type": "A", "position": 0, "distractor": "a0b"},
        ]
        path = tmp_path / "test.jsonl"
        self._write_distractors(path, distractors)
        with pytest.raises(ValueError, match="Duplicate"):
            validate_and_index_distractors(path)

    def test_incomplete_grid(self, tmp_path):
        """Raises ValueError listing missing pairs when grid is incomplete."""
        distractors = [
            {"id": "A_0", "type": "A", "position": 0, "distractor": "a0"},
            {"id": "A_1", "type": "A", "position": 1, "distractor": "a1"},
            {"id": "B_0", "type": "B", "position": 0, "distractor": "b0"},
            # missing B_1
        ]
        path = tmp_path / "test.jsonl"
        self._write_distractors(path, distractors)
        with pytest.raises(ValueError, match="do not form a full"):
            validate_and_index_distractors(path)


# ============================================================
# 2b. File I/O tests — write_injected_files
# ============================================================

class TestWriteInjectedFiles:
    def test_creates_correct_files(self, tmp_path):
        benchmark_rows = [{"question": "What is 2+2?", "expected_answer": "4"}]
        distractor_index = {
            ("A", "0"): {"id": "A_0", "distractor": "noise_a0"},
            ("B", "0"): {"id": "B_0", "distractor": "noise_b0"},
        }
        write_injected_files(
            benchmark_rows=benchmark_rows,
            benchmark="test",
            subset="sub",
            iteration=1,
            distractor_index=distractor_index,
            distractor_types=["A", "B"],
            positions=["0"],
            workdir=tmp_path,
        )
        iter_dir = tmp_path / "iter1"
        assert (iter_dir / "test_sub__type-A__pos-0.jsonl").exists()
        assert (iter_dir / "test_sub__type-B__pos-0.jsonl").exists()

    def test_injects_distractor_fields(self, tmp_path):
        benchmark_rows = [{"question": "Q1"}, {"question": "Q2"}]
        distractor_index = {
            ("A", "0"): {"id": "A_0", "distractor": "noise"},
        }
        write_injected_files(
            benchmark_rows=benchmark_rows,
            benchmark="test",
            subset="sub",
            iteration=1,
            distractor_index=distractor_index,
            distractor_types=["A"],
            positions=["0"],
            workdir=tmp_path,
        )
        out_file = tmp_path / "iter1" / "test_sub__type-A__pos-0.jsonl"
        rows = [json.loads(line) for line in open(out_file)]
        assert len(rows) == 2
        for row in rows:
            assert row["distractor_id"] == "A_0"
            assert row["distractor"] == "noise"
            assert row["question"] == rows[0]["question"] or row["question"] == "Q2"


# ============================================================
# 2c. File I/O tests — load_score
# ============================================================

class TestLoadScore:
    def _create_metrics(self, base_path, iteration_id, dtype, pos, metrics_data):
        """Helper to create metrics.json at the expected path."""
        metrics_dir = (
            base_path / "logs" / f"iter{iteration_id}" / "step3"
            / f"type-{dtype}_pos-{pos}" / "eval-results" / f"iter{iteration_id}"
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(metrics_data, f)

    def test_normal(self, tmp_path):
        self._create_metrics(tmp_path, "1", "RANDOM_FACT", "0", {
            "iter1": {"pass@1": {"num_entries": 198, "symbolic_correct": 58.5, "no_answer": 0.5}}
        })
        score = load_score(tmp_path, "1", "RANDOM_FACT", "0")
        assert score == 58.5

    def test_missing_file(self, tmp_path):
        score = load_score(tmp_path, "1", "RANDOM_FACT", "0")
        assert score is None

    def test_zero_entries(self, tmp_path):
        self._create_metrics(tmp_path, "1", "RANDOM_FACT", "0", {
            "iter1": {"pass@1": {"num_entries": 0, "symbolic_correct": 0.0, "no_answer": 0.0}}
        })
        score = load_score(tmp_path, "1", "RANDOM_FACT", "0")
        assert score is None

    def test_all_no_answer(self, tmp_path):
        self._create_metrics(tmp_path, "1", "RANDOM_FACT", "0", {
            "iter1": {"pass@1": {"num_entries": 198, "symbolic_correct": 0.0, "no_answer": 100.0}}
        })
        score = load_score(tmp_path, "1", "RANDOM_FACT", "0")
        assert score is None


# ============================================================
# 2d. File I/O tests — create/load/save state
# ============================================================

class TestStateIO:
    def test_create_and_load_initial_state(self, tmp_path):
        state_file = tmp_path / "current_state.jsonl"
        create_initial_state(state_file)
        state = load_state(state_file)
        assert len(state) == 15  # 5 types x 3 positions
        for item in state:
            assert item["parent_id"] is None
            assert item["parent_score"] is None
            assert "id" in item
            assert "type" in item
            assert "position" in item
            assert "distractor" in item

    def test_save_and_load_roundtrip(self, tmp_path):
        data = [
            {"id": "test_0", "type": "A", "position": 0, "distractor": "hello", "eval_score": 50.0},
            {"id": "test_1", "type": "B", "position": 1, "distractor": "world", "eval_score": 60.0},
        ]
        state_file = tmp_path / "state.jsonl"
        save_state(data, state_file)
        loaded = load_state(state_file)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "test_0"
        assert loaded[1]["eval_score"] == 60.0


# ============================================================
# 3. Mock-based tests — perflab_mutation_call
# ============================================================

class TestPerfLabMutationCall:
    @pytest.fixture(autouse=True)
    def setup_step1_globals(self):
        """Set up step1 module globals for testing."""
        import step1
        self._orig_client = step1.client
        self._orig_model = step1.MODEL
        step1.MODEL = "test-model"
        step1.client = MagicMock()
        yield
        step1.client = self._orig_client
        step1.MODEL = self._orig_model

    def _mock_response(self, content, finish_reason="stop"):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].finish_reason = finish_reason
        resp.choices[0].message.content = content
        return resp

    def test_returns_content(self):
        import step1
        step1.client.chat.completions.create.return_value = self._mock_response("mutated text")
        from step1 import perflab_mutation_call
        result = perflab_mutation_call(
            {"type": "RANDOM_FACT", "distractor": "original"},
            "CODE_SNIPPET", 1.0, 42
        )
        assert result == "mutated text"

    def test_strips_quotes(self):
        import step1
        step1.client.chat.completions.create.return_value = self._mock_response('"quoted result"')
        from step1 import perflab_mutation_call
        result = perflab_mutation_call(
            {"type": "RANDOM_FACT", "distractor": "original"},
            "CODE_SNIPPET", 1.0, 42
        )
        assert result == "quoted result"

    def test_retries_on_refusal(self):
        import step1
        refusal = self._mock_response(None, "refusal")
        success = self._mock_response("success after retry")
        step1.client.chat.completions.create.side_effect = [refusal, refusal, success]
        from step1 import perflab_mutation_call
        result = perflab_mutation_call(
            {"type": "RANDOM_FACT", "distractor": "original"},
            "CODE_SNIPPET", 1.0, 42
        )
        assert result == "success after retry"
        assert step1.client.chat.completions.create.call_count == 3

    def test_falls_back_on_all_refusals(self):
        import step1
        refusal = self._mock_response(None, "refusal")
        step1.client.chat.completions.create.return_value = refusal
        from step1 import perflab_mutation_call
        result = perflab_mutation_call(
            {"type": "RANDOM_FACT", "distractor": "original text"},
            "CODE_SNIPPET", 1.0, 42
        )
        assert result == "original text"  # falls back to original


# ============================================================
# 3b. Mock-based tests — _truncate_to_tokens
# ============================================================

class TestTruncateToTokens:
    def test_no_change_when_under_limit(self):
        import step1
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(30))  # 30 tokens
        step1._TRUNCATION_TOKENIZER = mock_tok
        try:
            result = step1._truncate_to_tokens("short text", 60)
            assert result == "short text"
            mock_tok.decode.assert_not_called()
        finally:
            step1._TRUNCATION_TOKENIZER = None

    def test_truncates_when_over_limit(self):
        import step1
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(200))  # 200 tokens
        mock_tok.decode.return_value = "truncated"
        step1._TRUNCATION_TOKENIZER = mock_tok
        try:
            result = step1._truncate_to_tokens("very long text " * 50, 60)
            assert result == "truncated"
            mock_tok.decode.assert_called_once_with(list(range(60)), skip_special_tokens=True)
        finally:
            step1._TRUNCATION_TOKENIZER = None


# ============================================================
# 3c. Mock-based tests — run_summarize_results
# ============================================================

class TestRunSummarizeResults:
    def test_skips_existing_metrics(self, tmp_path):
        """Doesn't call subprocess when metrics.json already exists."""
        dtype, pos = "RANDOM_FACT", "0"
        metrics_dir = tmp_path / "logs" / "iter1" / "step3" / f"type-{dtype}_pos-{pos}" / "eval-results" / "iter1"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text("{}")
        with patch("step4.subprocess.run") as mock_run:
            run_summarize_results(tmp_path, "1", [dtype], [pos])
        mock_run.assert_not_called()

    def test_skips_missing_output(self, tmp_path):
        """Skips when eval output files don't exist (eval failed/timed out)."""
        dtype, pos = "RANDOM_FACT", "0"
        # Don't create any output files
        with patch("step4.subprocess.run") as mock_run:
            run_summarize_results(tmp_path, "1", [dtype], [pos])
        mock_run.assert_not_called()

    def test_runs_when_output_exists(self, tmp_path):
        """Runs subprocess when output files present but no metrics.json."""
        dtype, pos = "RANDOM_FACT", "0"
        output_dir = tmp_path / "logs" / "iter1" / "step3" / f"type-{dtype}_pos-{pos}" / "eval-results" / "iter1"
        output_dir.mkdir(parents=True)
        (output_dir / "output-rs0.jsonl").write_text('{"test": true}\n')
        with patch("step4.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            run_summarize_results(tmp_path, "1", [dtype], [pos])
        mock_run.assert_called_once()


# ============================================================
# 4. Integration test — full step4 flow
# ============================================================

class TestStep4FullFlow:
    def test_end_to_end_selection(self, tmp_path):
        """Full step4 flow: summarize → load scores → select → write state."""
        from constants import DISTRACTOR_TYPES

        # Use 2 types x 2 positions = 4 slots for simplicity
        types = DISTRACTOR_TYPES[:2]  # RANDOM_FACT, CODE_SNIPPET
        positions = ["0", "1"]

        # Create archive with known parent_scores
        archive = []
        for t in types:
            for p in positions:
                archive.append({
                    "id": f"{t}_{p}_new",
                    "type": t,
                    "position": int(p),
                    "distractor": f"new distractor {t} {p}",
                    "parent_id": f"{t}_{p}_old",
                    "parent_score": 60.0,
                })
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        with open(archive_dir / "iter1.jsonl", "w") as f:
            for item in archive:
                f.write(json.dumps(item) + "\n")

        # Create current_state.jsonl (parent state)
        parent_state = []
        for t in types:
            for p in positions:
                parent_state.append({
                    "id": f"{t}_{p}_old",
                    "type": t,
                    "position": int(p),
                    "distractor": f"parent distractor {t} {p}",
                    "eval_score": 60.0,
                })
        with open(tmp_path / "current_state.jsonl", "w") as f:
            for item in parent_state:
                f.write(json.dumps(item) + "\n")

        # Create metrics.json — RANDOM_FACT_0: improved (50<60), RANDOM_FACT_1: worsened (70>60),
        # CODE_SNIPPET_0: missing (eval failed), CODE_SNIPPET_1: equal (60==60)
        scores = {
            ("RANDOM_FACT", "0"): 50.0,    # improved → keep
            ("RANDOM_FACT", "1"): 70.0,    # worsened → revert
            # CODE_SNIPPET_0: no metrics.json → revert
            ("CODE_SNIPPET", "1"): 60.0,   # equal → revert
        }
        for (t, p), score in scores.items():
            metrics_dir = (
                tmp_path / "logs" / "iter1" / "step3"
                / f"type-{t}_pos-{p}" / "eval-results" / "iter1"
            )
            metrics_dir.mkdir(parents=True)
            metrics = {"iter1": {"pass@1": {"num_entries": 198, "symbolic_correct": score, "no_answer": 0.5}}}
            with open(metrics_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

        # Run step4 logic (inline, not via main())
        from step4 import build_parent_index

        archive_data = read_jsonl(archive_dir / "iter1.jsonl")
        _positions = sorted(set(str(item["position"]) for item in archive_data))

        # Summarize (will skip CODE_SNIPPET_0 since no output exists)
        run_summarize_results(tmp_path, "1", types, _positions)

        # Load scores
        scored = []
        for item in archive_data:
            score = load_score(tmp_path, "1", str(item["type"]), str(item["position"]))
            scored_item = dict(item)
            scored_item["eval_score"] = score
            scored.append(scored_item)

        # Select
        parent_index = build_parent_index(tmp_path)
        selected = []
        for item in scored:
            p = parent_index.get((str(item["type"]), str(item["position"])))
            selected.append(select_distractor(item, p))

        # Verify
        by_slot = {(s["type"], str(s["position"])): s for s in selected}

        # RANDOM_FACT_0: score 50 < parent 60 → kept new
        assert by_slot[("RANDOM_FACT", "0")]["id"] == "RANDOM_FACT_0_new"

        # RANDOM_FACT_1: score 70 > parent 60 → reverted to parent
        assert by_slot[("RANDOM_FACT", "1")]["id"] == "RANDOM_FACT_1_old"

        # CODE_SNIPPET_0: no metrics → eval_score=None → reverted to parent
        assert by_slot[("CODE_SNIPPET", "0")]["id"] == "CODE_SNIPPET_0_old"

        # CODE_SNIPPET_1: score 60 == parent 60 → reverted to parent
        assert by_slot[("CODE_SNIPPET", "1")]["id"] == "CODE_SNIPPET_1_old"
