from __future__ import annotations

import csv
import subprocess
import sys
from io import StringIO
from pathlib import Path

from paper.scripts import build_adjudication_workbook as workbook


def test_parse_draft_accepts_the_dossier_recommendation_formats(tmp_path: Path) -> None:
    draft = tmp_path / "batch.md"
    draft.write_text(
        "\n".join(
            (
                "## C0001 - Example one",
                "- Recommendation: **included / include-1**. Exclusion reason: `NR`.",
                "### C0002 - Example two",
                "- **Recommendation:** **excluded - `exclude-out-of-scope`**.",
                "## C0003 - Example three",
                "- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `boundary` / `boundary`.",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows = workbook.parse_draft(draft)

    assert [row.candidate_id for row in rows] == ["C0001", "C0002", "C0003"]
    assert [(row.status, row.criterion) for row in rows] == [
        ("included", "include-1"),
        ("excluded", "exclude-out-of-scope"),
        ("boundary", "boundary"),
    ]
    assert [row.needs_accountable_author_review for row in rows] == [
        False,
        False,
        True,
    ]


def test_build_rows_binds_drafts_to_sealed_ratings_and_conflicts() -> None:
    recommendations = [
        workbook.DraftRecommendation(
            candidate_id="C0001",
            status="included",
            criterion="include-1",
            needs_accountable_author_review=False,
            draft_file="batch-01.md",
        ),
        workbook.DraftRecommendation(
            candidate_id="C0002",
            status="excluded",
            criterion="exclude-insufficient-detail",
            needs_accountable_author_review=True,
            draft_file="batch-01.md",
        ),
    ]
    ratings = {
        "C0001": (
            {
                "assignment_id": "A-C0001-01",
                "coder_id": "screening-01",
                "input_sha256": "a" * 64,
                "screening_status": "included",
                "criterion": "include-1",
                "access_status": "full_text",
                "exclusion_reason": "NR",
            },
            {
                "assignment_id": "A-C0001-02",
                "coder_id": "screening-02",
                "input_sha256": "a" * 64,
                "screening_status": "excluded",
                "criterion": "exclude-out-of-scope",
                "access_status": "abstract_only",
                "exclusion_reason": "NR",
            },
        ),
        "C0002": (
            {
                "assignment_id": "A-C0002-01",
                "coder_id": "screening-03",
                "input_sha256": "b" * 64,
                "screening_status": "excluded",
                "criterion": "exclude-insufficient-detail",
                "access_status": "abstract_only",
                "exclusion_reason": "No primary course method was inspectable.",
            },
            {
                "assignment_id": "A-C0002-02",
                "coder_id": "screening-04",
                "input_sha256": "b" * 64,
                "screening_status": "excluded",
                "criterion": "exclude-insufficient-detail",
                "access_status": "full_text",
                "exclusion_reason": "No primary course method was inspectable.",
            },
        ),
    }
    unresolved = {
        "C0002": ({"conflict_id": "X0002"},),
    }

    rows = workbook.build_rows(
        recommendations,
        ratings,
        unresolved,
        snapshot_sha256="c" * 64,
        primary_snapshot_sha256="d" * 64,
    )

    assert rows == [
        {
            "candidate_id": "C0001",
            "draft_file": "batch-01.md",
            "draft_status": "included",
            "draft_criterion": "include-1",
            "needs_accountable_author_review": "false",
            "input_sha256": "a" * 64,
            "snapshot_sha256": "c" * 64,
            "primary_snapshot_sha256": "d" * 64,
            "assignment_ids": "A-C0001-01;A-C0001-02",
            "reviewer_ids": "screening-01;screening-02",
            "raw_access_statuses": "full_text;abstract_only",
            "trigger_ids": "A1;A2",
            "resolved_conflict_ids": "NR",
            "conversion_status": "ready_for_normalization",
        },
        {
            "candidate_id": "C0002",
            "draft_file": "batch-01.md",
            "draft_status": "excluded",
            "draft_criterion": "exclude-insufficient-detail",
            "needs_accountable_author_review": "true",
            "input_sha256": "b" * 64,
            "snapshot_sha256": "c" * 64,
            "primary_snapshot_sha256": "d" * 64,
            "assignment_ids": "A-C0002-01;A-C0002-02",
            "reviewer_ids": "screening-03;screening-04",
            "raw_access_statuses": "abstract_only;full_text",
            "trigger_ids": "A4",
            "resolved_conflict_ids": "X0002",
            "conversion_status": "requires_accountable_author_review",
        },
    ]


def test_render_workbook_uses_a_fixed_header_and_lf_newlines() -> None:
    payload = workbook.render_workbook(
        [
            {
                field: field.upper()
                for field in workbook.WORKSHEET_HEADER
            }
        ]
    )

    assert "\r" not in payload
    rows = list(csv.DictReader(StringIO(payload)))
    assert rows == [
        {field: field.upper() for field in workbook.WORKSHEET_HEADER}
    ]


def test_direct_script_execution_can_render_help() -> None:
    script = Path(workbook.__file__)

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "non-final adjudication reconciliation worksheet" in completed.stdout
