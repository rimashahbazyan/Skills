import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


REQUIRED_DISTRACTOR_KEYS = {"id", "type", "position", "distractor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject distractors into benchmark JSONL files.")
    parser.add_argument("--iter", dest="iteration", type=int, required=True, help="Iteration index.")
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder where current_state.jsonl will be written.",
    )
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name.")
    parser.add_argument("--subset", type=str, required=True, help="Subset name.")
    return parser.parse_args()


def ensure_benchmark_file(benchmark: str, subset: str, workdir: Path) -> Path:
    local_file = workdir / f"{benchmark}_{subset}.jsonl"
    if local_file.exists():
        return local_file

    source_file = Path("/nemo_run/code/nemo_skills/dataset") / benchmark / f"{subset}.jsonl"
    if not source_file.exists():
        raise FileNotFoundError(
            f"Benchmark source file not found: {source_file}. "
            f"Cannot create local benchmark file {local_file}."
        )

    shutil.copy2(source_file, local_file)
    logging.info("Copied benchmark file from %s to %s", source_file, local_file)
    return local_file


def read_jsonl(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_and_index_distractors(iter_file: Path) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Dict]]:
    distractor_rows = read_jsonl(iter_file)
    if not distractor_rows:
        raise ValueError(f"No distractors found in {iter_file}.")

    indexed: Dict[Tuple[str, str], Dict] = {}
    types = set()
    positions = set()

    for row_num, row in enumerate(distractor_rows, start=1):
        missing = REQUIRED_DISTRACTOR_KEYS - row.keys()
        if missing:
            raise ValueError(f"Missing required keys {sorted(missing)} in {iter_file} line {row_num}.")

        distractor_type = str(row["type"])
        position = str(row["position"])
        key = (distractor_type, position)

        if key in indexed:
            raise ValueError(
                f"Expected exactly one distractor per (type, position). Duplicate found for {key} in {iter_file}."
            )

        indexed[key] = row
        types.add(distractor_type)
        positions.add(position)

    sorted_types = sorted(types)
    sorted_positions = sorted(positions)
    expected_count = len(sorted_types) * len(sorted_positions)
    if len(indexed) != expected_count:
        missing_pairs = [
            (d_type, pos)
            for d_type in sorted_types
            for pos in sorted_positions
            if (d_type, pos) not in indexed
        ]
        raise ValueError(
            "Distractors do not form a full type-position grid. "
            f"Found={len(indexed)}, expected={expected_count}, missing={missing_pairs}."
        )

    return sorted_types, sorted_positions, indexed


def write_injected_files(
    benchmark_rows: List[Dict],
    benchmark: str,
    subset: str,
    iteration: int,
    distractor_index: Dict[Tuple[str, str], Dict],
    distractor_types: List[str],
    positions: List[str],
    workdir: Path,
) -> None:
    out_dir = workdir / f"iter{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files = 0
    for distractor_type in distractor_types:
        for position in positions:
            pair = (distractor_type, position)
            distractor_row = distractor_index[pair]

            out_file = out_dir / (
                f"{benchmark}_{subset}__type-{distractor_type}__pos-{position}.jsonl"
            )
            with open(out_file, "w", encoding="utf-8") as f:
                for row in benchmark_rows:
                    out_row = dict(row)
                    out_row["distractor_id"] = distractor_row["id"]
                    out_row["distractor"] = distractor_row["distractor"]
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            written_files += 1

    expected_files = len(distractor_types) * len(positions)
    if written_files != expected_files:
        raise RuntimeError(f"Expected to write {expected_files} files, but wrote {written_files}.")

    logging.info("Wrote %d benchmark files to %s", written_files, out_dir)


def main() -> None:
    args = parse_args()
    workdir = Path(args.output_folder)
    benchmark_file = ensure_benchmark_file(args.benchmark, args.subset, workdir)
    iter_file = workdir / "archive" / f"iter{args.iteration}.jsonl"
    if not iter_file.exists():
        raise FileNotFoundError(f"Distractor file not found: {iter_file}")

    benchmark_rows = read_jsonl(benchmark_file)
    distractor_types, positions, distractor_index = validate_and_index_distractors(iter_file)

    write_injected_files(
        benchmark_rows=benchmark_rows,
        benchmark=args.benchmark,
        subset=args.subset,
        iteration=args.iteration,
        distractor_index=distractor_index,
        distractor_types=distractor_types,
        positions=positions,
        workdir=workdir,
    )

    logging.info(
        "Step2 completed. Grid size: %d types x %d positions = %d files.",
        len(distractor_types),
        len(positions),
        len(distractor_types) * len(positions),
    )

if __name__ == "__main__":
    main()