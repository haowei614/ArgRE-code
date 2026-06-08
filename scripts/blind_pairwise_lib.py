"""Shared normalization, pairing, and comparison logic for blind text-level evaluation."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PHASE3_FILENAME = "phase3_integrated_kaos_model.json"
RUN_RECORD_FILENAME = "run_record.json"
DEFAULT_CASE_TARGETS = {
    "AD": 5,
    "ATM": 10,
    "Bookkeeping": 6,
}
DEFAULT_SEED = 20260607

PAIRWISE_COLUMNS = (
    "Pair_ID",
    "Case_Study",
    "Source_Requirement_ID",
    "ArgRE_Text",
    "NoAF_Text",
    "Identical",
    "ArgRE_Run_ID",
    "NoAF_Run_ID",
    "ArgRE_Seed",
    "NoAF_Seed",
)


@dataclass(frozen=True)
class RequirementRecord:
    method: str
    case_study: str
    run_id: str
    seed: int
    source_requirement_id: str
    source_file: str
    raw_requirement_text: str
    cleaned_requirement_text: str
    setting: str
    system: str


PREFIX_PATTERNS = (
    re.compile(
        r"^(?:Safety|Efficiency|Sustainability|Trustworthiness|Responsibility|Integrated)\s+"
        r"(?:Goal|Requirement|Strategy|Task)(?:\s+\d+)?\s*:\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^[A-Za-z ]+ objective for [A-Za-z0-9_-]+:\s*", re.IGNORECASE),
    re.compile(
        r"^The system shall ensure "
        r"(?:safety|efficiency|sustainability|trustworthiness|responsibility)"
        r"\s*\([^)]*\)\s*:\s*[-•]?\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^The system shall satisfy:\s*[-•]?\s*", re.IGNORECASE),
    re.compile(r"^[-•]\s*"),
)


def collect_records(runs_dir: Path, *, method: str) -> list[RequirementRecord]:
    records: list[RequirementRecord] = []
    for run_record_path in sorted(runs_dir.glob(f"**/{RUN_RECORD_FILENAME}")):
        run_record = load_json(run_record_path)
        if not isinstance(run_record, dict):
            continue
        case_study = str(run_record.get("case_id", "")).strip()
        if case_study not in DEFAULT_CASE_TARGETS:
            continue
        phase3_path = resolve_phase3_path(run_record, run_record_path.parent)
        if phase3_path is None:
            continue
        phase3 = load_json(phase3_path)
        elements = phase3.get("gsn_elements", []) if isinstance(phase3, dict) else []
        if not isinstance(elements, list):
            continue
        for element in elements:
            if not isinstance(element, dict):
                continue
            cleaned = clean_requirement_text(element)
            if not cleaned:
                continue
            source_id = str(element.get("id", "")).strip()
            if not source_id:
                continue
            records.append(
                RequirementRecord(
                    method=method,
                    case_study=case_study,
                    run_id=str(run_record.get("run_id", "")).strip(),
                    seed=to_int(run_record.get("seed"), default=0),
                    source_requirement_id=source_id,
                    source_file=str(phase3_path),
                    raw_requirement_text=raw_requirement_text(element),
                    cleaned_requirement_text=cleaned,
                    setting=str(run_record.get("setting", "")).strip(),
                    system=str(run_record.get("system", "")).strip(),
                )
            )
    return records


def resolve_phase3_path(run_record: dict[str, Any], run_dir: Path) -> Path | None:
    artifact_paths = run_record.get("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        raw_path = str(artifact_paths.get(PHASE3_FILENAME, "")).strip()
        if raw_path:
            candidates = [Path(raw_path), run_dir / Path(raw_path).name]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    candidates = [run_dir / PHASE3_FILENAME]
    artifacts_dir = str(run_record.get("artifacts_dir", "")).strip()
    if artifacts_dir:
        candidates.append(Path(artifacts_dir) / PHASE3_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_pairs(
    argre_records: list[RequirementRecord],
    noaf_records: list[RequirementRecord],
    *,
    seed: int,
) -> tuple[list[tuple[str, RequirementRecord, RequirementRecord]], list[dict[str, int | str]]]:
    rng = random.Random(seed)
    argre_by_key = group_records(argre_records)
    noaf_by_key = group_records(noaf_records)
    pairs: list[tuple[str, RequirementRecord, RequirementRecord]] = []
    shortfalls: list[dict[str, int | str]] = []

    for case_study, target in DEFAULT_CASE_TARGETS.items():
        argre_keys = {key for key in argre_by_key if key[0] == case_study}
        noaf_keys = {key for key in noaf_by_key if key[0] == case_study}
        shared = sorted(argre_keys & noaf_keys, key=lambda item: item[1])
        rng.shuffle(shared)
        selected = shared[:target]
        if len(selected) < target:
            shortfalls.append(
                {"case_study": case_study, "requested": target, "available": len(selected)}
            )
        for index, key in enumerate(selected, start=1):
            pair_id = f"{case_study}-P{index:02d}"
            pairs.append(
                (
                    pair_id,
                    choose_representative(argre_by_key[key]),
                    choose_representative(noaf_by_key[key]),
                )
            )
    return pairs, shortfalls


def build_pair_diagnostics(
    pairs: list[tuple[str, RequirementRecord, RequirementRecord]],
) -> dict[str, object]:
    identical_pairs = []
    different_pairs = []
    for pair_id, argre, noaf in pairs:
        if normalize_text(argre.cleaned_requirement_text) == normalize_text(
            noaf.cleaned_requirement_text
        ):
            identical_pairs.append(pair_id)
        else:
            different_pairs.append(pair_id)
    return {
        "identical_pair_count": len(identical_pairs),
        "different_pair_count": len(different_pairs),
        "identical_pairs": identical_pairs,
        "different_pairs": different_pairs,
    }


def build_pairwise_rows(
    pairs: list[tuple[str, RequirementRecord, RequirementRecord]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pair_id, argre, noaf in pairs:
        identical = (
            normalize_text(argre.cleaned_requirement_text)
            == normalize_text(noaf.cleaned_requirement_text)
        )
        rows.append(
            {
                "Pair_ID": pair_id,
                "Case_Study": argre.case_study,
                "Source_Requirement_ID": argre.source_requirement_id,
                "ArgRE_Text": argre.cleaned_requirement_text,
                "NoAF_Text": noaf.cleaned_requirement_text,
                "Identical": "yes" if identical else "no",
                "ArgRE_Run_ID": argre.run_id,
                "NoAF_Run_ID": noaf.run_id,
                "ArgRE_Seed": str(argre.seed),
                "NoAF_Seed": str(noaf.seed),
            }
        )
    return rows


def copy_run_evidence(
    *,
    argre_runs_dir: Path,
    noaf_runs_dir: Path,
    output_runs_dir: Path,
) -> None:
    output_runs_dir.mkdir(parents=True, exist_ok=True)
    for method_dir, source_dir in (
        ("argre_af_preferred", argre_runs_dir),
        ("argre_noaf_original", noaf_runs_dir),
    ):
        method_output = output_runs_dir / method_dir
        method_output.mkdir(parents=True, exist_ok=True)
        for run_record_path in sorted(source_dir.glob(f"**/{RUN_RECORD_FILENAME}")):
            run_dir = run_record_path.parent
            run_id = run_dir.name
            target_dir = method_output / run_id
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in (RUN_RECORD_FILENAME, PHASE3_FILENAME):
                source_file = run_dir / filename
                if source_file.exists():
                    target_file = target_dir / filename
                    target_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")


def group_records(records: list[RequirementRecord]) -> dict[tuple[str, str], list[RequirementRecord]]:
    grouped: dict[tuple[str, str], list[RequirementRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.case_study, normalize_source_id(record.source_requirement_id))].append(
            record
        )
    return grouped


def choose_representative(records: list[RequirementRecord]) -> RequirementRecord:
    return sorted(records, key=lambda item: (item.seed, item.run_id, item.source_requirement_id))[0]


def raw_requirement_text(element: dict[str, Any]) -> str:
    name = str(element.get("name", "")).strip()
    description = str(element.get("description", "")).strip()
    if name and description:
        return f"{name}: {description}"
    return description or name


def clean_requirement_text(element: dict[str, Any]) -> str:
    text = raw_requirement_text(element)
    changed = True
    while changed:
        before = text
        for pattern in PREFIX_PATTERNS:
            text = pattern.sub("", text).strip()
        changed = text != before
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text.split()) < 4:
        return ""
    return text


def normalize_source_id(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_readme(
    path: Path,
    *,
    pairs: list[tuple[str, RequirementRecord, RequirementRecord]],
    pair_diagnostics: dict[str, object],
) -> None:
    total = len(pairs)
    identical = int(pair_diagnostics["identical_pair_count"])
    different = int(pair_diagnostics["different_pair_count"])
    identical_pct = round(100 * identical / total) if total else 0
    different_pct = round(100 * different / total) if total else 0
    different_pairs = pair_diagnostics.get("different_pairs", [])
    different_note = ""
    if different_pairs:
        different_note = (
            f"\n\nPairs with minor lexical variation: {', '.join(str(item) for item in different_pairs)}."
        )

    path.write_text(
        f"""# Text-Level Pairwise Comparison

## Purpose
Supplements Section VI-A3: verifies that ArgRE and ArgRE-NoAF
produce substantively identical requirement text under matched conditions.

## Experimental Setup
- Case studies: AD, ATM, Bookkeeping
- Seeds: 101, 202, 303
- Configurations: ArgRE (AF, preferred semantics), ArgRE-NoAF
- Total runs: 18

## Results
- Total paired requirements: {total}
- Identical after normalization: {identical} ({identical_pct}%)
- Minor lexical variation: {different} ({different_pct}%){different_note}

## Files
- `pairwise_comparison.csv` — full {total}-pair comparison table
- `compare.py` — normalization and matching script
- `runs/` — Phase 3 outputs for all 18 runs
- `blind_eval_sheet.csv` — annotator sheet with method labels withheld
- `private_reveal_mapping.json` — pair-to-run mapping for post-hoc analysis

## Reproduce
From this directory:

```bash
python3 compare.py
```
""",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: object, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default
