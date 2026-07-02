from __future__ import annotations

import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "paper" / "data"
PROTOCOL_PATH = DATA_ROOT / "screening_protocol.md"
PROMPT_PATH = DATA_ROOT / "screening_reviewer_prompt.md"
TAXONOMY_PATH = DATA_ROOT / "taxonomy.json"
V7_ROOT = DATA_ROOT / "screening_work" / "v7"
RESULT_HEADER = (
    "assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,"
    "coder_id,screened_on,screening_status,criterion,access_status,source_urls,"
    "evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,"
    "screening_locator,exclusion_reason,notes"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_read(path).split())


def test_v7_root_and_working_contracts_are_synchronized() -> None:
    assert _read(PROTOCOL_PATH) == _read(V7_ROOT / "protocol.md")
    assert _read(PROMPT_PATH) == _read(V7_ROOT / "reviewer_prompt_template.md")


def test_v7_uses_binary_final_statuses_and_historical_only_boundary() -> None:
    text = _normalized(PROTOCOL_PATH)
    assert "Final v7 screening statuses are exactly `included` and `excluded`." in text
    assert "`boundary` is historical terminology only and MUST NOT be assigned" in text
    assert not re.search(r"\| `boundary` \|", _read(PROTOCOL_PATH))


def test_v7_pairs_included_with_include_relevant_only() -> None:
    text = _read(PROTOCOL_PATH)
    assert "| `included` | `include-relevant` | `NR` |" in text
    assert "| `excluded` | Exactly one controlled exclusion criterion" in text
    assert "screening_inclusion_criterion is exactly [`include-relevant`]" in text


def test_v7_retains_fixed_routes_as_supporting_evidence_without_calling_them_methods() -> None:
    text = _normalized(PROTOCOL_PATH)
    assert "Fixed CARLA routes or an equivalent fixed-route source are retained" in text
    assert "citable representation, benchmark format, simulator interface, or evaluation requirement" in text
    assert "MUST NOT be called a generation method" in text


def test_v7_pass_one_does_not_code_or_rank_contributions() -> None:
    text = _normalized(PROTOCOL_PATH)
    assert "Pass 1 MUST NOT choose or rank a primary contribution" in text
    assert "MUST NOT perform full Pass 2 coding" in text
    assert "Retained-source count is not a method count." in text


def test_v7_duplicate_reviewers_rate_the_same_frozen_evidence_packet() -> None:
    text = _normalized(PROTOCOL_PATH)
    assert "Both duplicate reviewers MUST rate the same frozen evidence packet." in text
    assert "MAY verify metadata or report a packet defect but MUST NOT silently replace or add eligibility evidence" in text
    assert "Stronger evidence discovered after freeze requires a new packet version." in text


def test_v7_pass_two_tiers_limit_supported_claims() -> None:
    text = _normalized(PROTOCOL_PATH)
    assert "Pass 2 is multi-label and separately assigns `survey_evidence_tier` as `core`, `supporting`, or `contextual`." in text
    assert "Supporting evidence MUST NOT substantiate generation-method claims." in text
    assert "Contextual evidence MUST NOT support implementation or performance claims." in text


def test_v7_result_csv_header_is_unchanged() -> None:
    for path in (PROTOCOL_PATH, PROMPT_PATH):
        blocks = re.findall(r"```csv\n([^`]+)\n```", _read(path))
        assert RESULT_HEADER in blocks


def test_v7_taxonomy_has_binary_result_statuses_and_preserves_corpus_statuses() -> None:
    taxonomy = json.loads(_read(TAXONOMY_PATH))
    assert taxonomy["screening_result_status"] == ["included", "excluded"]
    assert taxonomy["screening_status"] == ["candidate", "included", "excluded", "boundary"]
    assert taxonomy["screening_inclusion_criterion"] == ["include-relevant"]


def test_v7_working_readme_is_unfrozen_and_calibration_is_fresh() -> None:
    readme = _normalized(V7_ROOT / "README.md")
    assert "v7 is a working, unfrozen protocol." in readme
    assert "The evidence inventory MUST be complete before freeze or reviewer launch." in readme
    assert "No v7 main release exists." in readme

    text = _normalized(PROTOCOL_PATH)
    assert "fresh stable-30 calibration" in text
    assert "six blind reviewer contexts" in text
    assert "agreement >= 0.80" in text
    assert "no systematic ambiguity" in text
    assert "60 valid ratings" in text
    assert "MUST NOT receive v3-v6 ratings or disagreements" in text
