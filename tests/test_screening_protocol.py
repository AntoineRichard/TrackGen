from __future__ import annotations

import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "paper" / "data" / "screening_protocol.md"
TAXONOMY_PATH = REPOSITORY_ROOT / "paper" / "data" / "taxonomy.json"
README_PATH = REPOSITORY_ROOT / "paper" / "data" / "README.md"
REVIEWER_PROMPT_PATH = (
    REPOSITORY_ROOT / "paper" / "data" / "screening_reviewer_prompt.md"
)
V7_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "paper" / "data" / "screening_work" / "v7" / "protocol.md"
)
V7_REVIEWER_PROMPT_PATH = (
    REPOSITORY_ROOT
    / "paper"
    / "data"
    / "screening_work"
    / "v7"
    / "reviewer_prompt_template.md"
)
FULL_CACHE_ISOLATION_STATEMENT = (
    "Fresh context; no shared conversation history, memory, ratings, results, "
    "or retrieval cache."
)
LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT = (
    "Fresh context; no shared conversation history, memory, ratings, or "
    "results were supplied; provider retrieval-cache isolation was not exposed."
)
PROVIDER_METADATA_LIMITATION_KEYS = (
    "backend_model_version",
    "decoding_parameters",
    "developer_instruction_bytes",
    "retrieval_cache_isolation",
    "system_instruction_bytes",
)
CALIBRATION_SELECTION_HEADER = ("candidate_id",)

RESULT_HEADER = (
    "assignment_id",
    "phase",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
    "coder_id",
    "screened_on",
    "screening_status",
    "criterion",
    "access_status",
    "source_urls",
    "evidence_version",
    "evidence_retrieved_on",
    "evidence_archive_url",
    "evidence_sha256",
    "screening_locator",
    "exclusion_reason",
    "notes",
)
ADJUDICATION_HEADER = (
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "primary_snapshot_sha256",
    "assignment_ids",
    "adjudicator_id",
    "reviewer_ids",
    "decided_on",
    "screening_status",
    "criterion",
    "access_status",
    "source_urls",
    "evidence_version",
    "evidence_retrieved_on",
    "evidence_archive_url",
    "evidence_sha256",
    "screening_locator",
    "exclusion_reason",
    "resolution_evidence",
    "resolved_conflict_ids",
    "notes",
)
CALIBRATION_DECISION_HEADER = (
    "decision_id",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "candidate_ids_sha256",
    "assignment_ids_sha256",
    "status_agreement_numerator",
    "status_agreement_denominator",
    "status_agreement",
    "systematic_ambiguity",
    "decision",
    "decided_on",
    "decision_makers",
    "resolution_evidence",
)

CALIBRATION_DECISION_MANIFEST_HEADER = (
    "manifest_version",
    "calibration_decision_snapshot_sha256",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "decision_id",
    "decision_file_sha256",
    "candidate_ids_file_sha256",
    "assignment_ids_file_sha256",
    "row_count",
)
RELEASE_MANIFEST_HEADER = (
    "manifest_version",
    "phase",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "assignment_count",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
)
PHASE_RESULT_MANIFEST_HEADER = (
    "manifest_version",
    "phase_result_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "reviewer_release_sha256",
    "phase",
    "batch_id",
    "coder_id",
    "result_filename",
    "result_file_sha256",
    "row_count",
)

ADJUDICATION_MANIFEST_HEADER = (
    "manifest_version",
    "adjudication_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
    "main_result_snapshot_sha256",
    "primary_snapshot_sha256",
    "adjudication_file_sha256",
    "execution_registry_sha256",
    "row_count",
    "execution_row_count",
)
CITATION_KEY_HEADER = ("candidate_id", "cite_key")
AUTHOR_VERIFICATION_HEADER = (
    "candidate_id",
    "primary_snapshot_sha256",
    "adjudication_snapshot_sha256",
    "decision_sha256",
    "evidence_versions_sha256",
    "deciding_locators_sha256",
    "verified_by",
    "verified_role",
    "verified_on",
    "verification_status",
    "verification_evidence",
)
PROJECTION_MANIFEST_HEADER = (
    "manifest_version",
    "projection_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
    "main_result_snapshot_sha256",
    "primary_snapshot_sha256",
    "adjudication_snapshot_sha256",
    "execution_registry_sha256",
    "citation_key_ledger_sha256",
    "author_verification_sha256",
    "candidates_sha256",
    "citation_keys_sha256",
    "conflicts_sha256",
    "screening_decisions_sha256",
    "screening_agreement_sha256",
    "candidate_count",
    "decision_row_count",
    "agreement_row_count",
)

EXECUTION_REGISTER_HEADER = (
    "execution_id",
    "role_id",
    "role_type",
    "context_id",
    "task",
    "work_item_id",
    "model_identifier",
    "model_version",
    "configuration_sha256",
    "prompt_sha256",
    "provider",
    "runtime",
    "tool_configuration",
    "retrieval_configuration",
    "decoding_parameters",
    "system_instruction_sha256",
    "developer_instruction_sha256",
    "user_instruction_sha256",
    "cache_isolation_statement",
    "started_on",
    "completed_on",
    "result_file_sha256",
    "human_role",
    "training_calibration_exposure",
    "automated_actions",
)

SCREENING_STATUSES = {"included", "excluded"}
INCLUSION_CRITERIA = {"include-relevant"}
EXCLUSION_CRITERIA = {
    "exclude-fixed-racing-line",
    "exclude-appearance-dynamics",
    "exclude-traffic-only",
    "exclude-insufficient-detail",
    "exclude-out-of-scope",
}
ACCESS_STATUSES = {
    "full_text",
    "full_text_and_supplement",
    "official_documentation",
    "abstract_only",
}


def _protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


def _section(text: str, title: str, *, level: int) -> str:
    lines = text.splitlines()
    marker = f"{'#' * level} {title}"
    starts = [index for index, line in enumerate(lines) if line == marker]
    assert len(starts) == 1, f"expected one {marker!r} section, found {len(starts)}"

    start = starts[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        heading = re.fullmatch(r"(#+)\s+.+", lines[index])
        if heading and len(heading.group(1)) <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _table(section: str, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    table_lines = [line for line in section.splitlines() if line.startswith("|")]
    assert len(table_lines) >= 2, "section has no Markdown table"

    def cells(line: str) -> tuple[str, ...]:
        return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))

    assert cells(table_lines[0]) == expected_header
    separators = cells(table_lines[1])
    assert len(separators) == len(expected_header)
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = cells(line)
        assert len(values) == len(expected_header)
        rows.append(dict(zip(expected_header, values, strict=True)))
    return rows


def _unquote(value: str) -> str:
    match = re.fullmatch(r"`([^`]+)`", value)
    assert match, f"controlled value must be one complete code span: {value!r}"
    return match.group(1)


def _ordered_labels(section: str) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for line in section.splitlines()
        if (match := re.match(r"^\d+\. \*\*([^*]+)\.\*\*", line))
    )


def _normalized(section: str) -> str:
    return " ".join(section.replace("`", "").split())


def _csv_header(section: str) -> tuple[str, ...]:
    blocks = re.findall(r"```csv\n([^`]+)\n```", section)
    assert len(blocks) == 1, f"expected one CSV block, found {len(blocks)}"
    lines = blocks[0].splitlines()
    assert len(lines) == 1, "schema block must contain exactly one header"
    return tuple(lines[0].split(","))


def test_scope_definitions_and_review_unit_are_operationally_closed() -> None:
    text = _protocol()
    definitions = _table(
        _section(text, "Operational definitions", level=2),
        ("Term", "Operational definition"),
    )
    assert tuple(_unquote(row["Term"]) for row in definitions) == (
        "course",
        "racing robot",
        "transferable adjacent domain",
        "source-native contribution",
        "supporting transfer",
        "material evidence",
        "report",
        "work",
    )
    assert all("MUST" in row["Operational definition"] for row in definitions)

    policy = _normalized(_section(text, "Eligibility dimensions and synthesis unit", level=2))
    assert "No language restriction applies." in policy
    assert "No publication-date restriction applies." in policy
    assert "No publication-type restriction applies." in policy
    assert "The screening unit is the report; the synthesis unit is the work." in policy
    assert "Exact duplicate reports MUST be collapsed before assignment." in policy
    assert (
        "Versioned, companion, or overlapping reports MUST remain linked until full-text "
        "screening establishes whether each contains unique material evidence." in policy
    )
    assert "A work MUST be counted at most once in quantitative synthesis." in policy


def test_result_schema_is_exact_and_every_field_has_a_contract() -> None:
    text = _protocol()
    section = _section(text, "Reviewer result schema", level=2)
    assert _csv_header(section) == RESULT_HEADER

    field_rows = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in field_rows) == RESULT_HEADER
    assert all(row["Normative rule"] for row in field_rows)

    normalized = _normalized(section)
    assert "Every field MUST contain a value; blank fields are invalid." in normalized
    assert "screened_on MUST be an ISO 8601 calendar date in YYYY-MM-DD form." in normalized
    assert (
        "evidence_version and evidence_retrieved_on are REQUIRED for every rating."
        in normalized
    )
    assert (
        "source_urls MUST contain canonical, semicolon-separated absolute HTTP(S) URLs"
        in normalized
    )
    assert (
        "screening_locator MUST identify a page, section, table, figure, algorithm, "
        "appendix, or stable official-documentation anchor." in normalized
    )
    assert "notes MAY be NR" in normalized


def test_controlled_vocabularies_are_exact_and_closed() -> None:
    text = _protocol()
    vocabularies = (
        ("Screening statuses", SCREENING_STATUSES),
        ("Inclusion criteria", INCLUSION_CRITERIA),
        ("Exclusion criteria", EXCLUSION_CRITERIA),
        ("Access statuses", ACCESS_STATUSES),
    )
    for heading, expected in vocabularies:
        rows = _table(
            _section(text, heading, level=3),
            ("Value", "Normative meaning"),
        )
        actual = {_unquote(row["Value"]) for row in rows}
        assert actual == expected, heading
        assert len(rows) == len(expected), f"{heading} contains duplicate values"
        assert all(row["Normative meaning"] for row in rows)


def test_inclusion_covers_core_supporting_and_contextual_retention() -> None:
    text = _protocol()
    criteria = {
        _unquote(row["Value"]): _normalized(row["Normative meaning"])
        for row in _table(
            _section(text, "Inclusion criteria", level=3),
            ("Value", "Normative meaning"),
        )
    }
    assert criteria == {
        "include-relevant": (
            "Frozen packet evidence establishes at least one core, supporting, or "
            "contextual retention condition."
        )
    }
    assert not re.search(r"include-[1234]", text)
    assert "Pass 1 MUST NOT choose or rank a primary contribution" in text
    assert "MUST NOT perform full Pass 2 coding" in _normalized(text)
    assert "Retained-source count is not method count" in text

    procedure = _normalized(_section(text, "Normative decision procedure", level=2))
    assert (
        "Frozen packet evidence MUST establish at least one core, supporting, or "
        "contextual retention condition." in procedure
    )

    clarification = _normalized(
        _section(text, "Eligibility and supporting-transfer clarification", level=3)
    )
    assert "source-native script, implementation, algorithm, or specification" in clarification
    assert "reusable course representation" in clarification
    assert "Fixed CARLA routes or equivalent fixed-route sources" in clarification
    assert "MUST NOT be called generation methods" in clarification
    assert "generic laboratory, project, promotional, or topic page" in clarification
    assert "the source is excluded" in clarification


def test_decision_procedure_has_explicit_precedence_and_status_pairing() -> None:
    text = _protocol()
    procedure = _section(text, "Normative decision procedure", level=2)
    assert _ordered_labels(procedure) == (
        "Establish access",
        "Apply retention",
        "Apply exclusion",
        "Validate the result",
    )

    normalized = _normalized(procedure)
    assert "Retention has precedence over exclusion." in normalized
    assert "boundary MUST NOT be assigned as a v7 result or criterion" in normalized

    pairing = _table(
        _section(text, "Status and criterion pairing", level=3),
        ("screening_status", "Allowed criterion", "exclusion_reason"),
    )
    assert pairing == [
        {
            "screening_status": "`included`",
            "Allowed criterion": "`include-relevant`",
            "exclusion_reason": "`NR`",
        },
        {
            "screening_status": "`excluded`",
            "Allowed criterion": "Exactly one controlled exclusion criterion",
            "exclusion_reason": "A substantive, source-specific reason",
        },
    ]


def test_root_taxonomy_declares_v7_result_and_inclusion_vocabularies() -> None:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert taxonomy["screening_result_status"] == ["included", "excluded"]
    assert taxonomy["screening_inclusion_criterion"] == ["include-relevant"]
    assert taxonomy["screening_status"] == [
        "candidate",
        "included",
        "excluded",
        "boundary",
    ]


def test_access_gate_and_evidence_provenance_are_executable() -> None:
    text = _protocol()
    section = _normalized(_section(text, "Access and evidence rules", level=2))
    assert (
        "included MUST NOT use abstract_only; it requires full_text, "
        "full_text_and_supplement, or official_documentation." in section
    )
    assert (
        "Before assigning abstract_only in the evidence inventory, the coordinator MUST make and document an "
        "exhaustive retrieval attempt." in section
    )
    assert (
        "If only an abstract remains available and it cannot support a survey claim, "
        "the result MUST be excluded with criterion exclude-insufficient-detail." in section
    )
    assert (
        "An included rating based on official_documentation MUST record "
        "either a version-pinned evidence_archive_url or a lowercase 64-hex evidence_sha256."
        in section
    )
    assert (
        "evidence_archive_url MAY be NR only when no version-pinned archive was inspected."
        in section
    )
    assert (
        "evidence_sha256 MAY be NR only when the exact inspected bytes were not lawfully "
        "or technically obtainable." in section
    )
    assert (
        "Only evidence_archive_url, evidence_sha256, exclusion_reason, notes, and "
        "resolved_conflict_ids may contain NR under their field-specific rules." in section
    )


def test_calibration_is_a_versioned_release_gate_not_an_analysis_label() -> None:
    text = _protocol()
    section = _section(text, "Calibration and release gate", level=2)
    requirements = _table(
        _section(text, "Calibration gate requirements", level=3),
        ("Property", "Required value"),
    )
    assert requirements == [
        {"Property": "Calibration records", "Required value": "30"},
        {"Property": "Main records", "Required value": "172"},
        {"Property": "Final-version records", "Required value": "202"},
        {"Property": "Locked blind ratings per record", "Required value": "2"},
        {
            "Property": "Minimum calibration exact status agreement",
            "Required value": ">= 0.80",
        },
        {"Property": "Systematic-ambiguity tolerance", "Required value": "None"},
    ]
    assert _ordered_labels(section) == (
        "Release calibration",
        "Lock blind ratings",
        "Analyze agreement",
        "Discuss disagreements",
        "Record the gate decision",
        "Revise when required",
        "Release the main phase",
        "Complete final-version duplication",
    )

    normalized = _normalized(section)
    assert (
        "The stable 30-record calibration set MUST be released separately as phase calibration."
        in normalized
    )
    assert (
        "The main 172 records MUST NOT be released until an immutable calibration "
        "decision records both exact status agreement >= 0.80 and no systematic ambiguity."
        in normalized
    )
    assert (
        "Any substantive protocol revision MUST invalidate the calibration run, increment "
        "both the protocol version and snapshot version, and require fresh isolated reviewers "
        "to rerate the same stable 30 calibration records blindly." in normalized
    )
    assert (
        "Every one of the 202 records MUST ultimately have two locked ratings made under "
        "the same final protocol and snapshot version." in normalized
    )
    assert (
        "Calibration is a mandatory prospective workflow phase and release gate. "
        "It is not a post hoc subgroup or an analysis label." in normalized
    )


def test_calibration_selection_is_stable_stratified_and_status_blind() -> None:
    text = _protocol()
    section = _section(text, "Deterministic calibration selection", level=3)
    normalized = _normalized(section)
    strata = _table(section, ("Priority", "Coarse source-type stratum", "Matching tokens"))
    assert _csv_header(section) == CALIBRATION_SELECTION_HEADER
    assert tuple(row["Priority"] for row in strata) == ("1", "2", "3", "4", "5", "6")
    assert tuple(row["Coarse source-type stratum"] for row in strata) == (
        "`standard-specification`",
        "`competition`",
        "`benchmark-dataset`",
        "`software`",
        "`scholarly`",
        "`official-other`",
    )
    assert "trackgen-screening-calibration-v1" in normalized
    assert "MUST NOT depend on the protocol hash or protocol version." in normalized
    assert (
        "Only candidate_id, source_type, discovery_stream, and discovery_query may "
        "influence selection." in normalized
    )
    assert (
        "citation key, prior screening status, exclusion reason, conflict state, and "
        "metadata evidence must not influence selection." in normalized.lower()
    )
    assert "allocate an initial quota of min(2, stratum size)" in normalized
    assert "remaining places in proportion" in normalized
    assert "fill its quota greedily" in normalized
    assert "Unicode NFKC" in normalized
    assert "SHA-256(salt + NUL + candidate_id)" in normalized
    assert (
        "calibration_selection.csv MUST be a coordinator-root CSV with exactly "
        "the one-column header candidate_id and exactly 30 data rows." in normalized
    )
    assert (
        "Its row order MUST be SHA-256(salt + NUL + candidate_id), then UTF-8 "
        "candidate_id; the file is covered by SHA256SUMS." in normalized
    )
    assert (
        "The 30 candidate IDs in calibration_selection.csv MUST equal exactly the "
        "candidate IDs whose two manifest rows have phase calibration." in normalized
    )
    assert (
        "Any corpus other than exactly 202 unique candidate IDs is a hard error."
        in normalized
    )


def test_calibration_decision_schema_and_release_controls_are_exact() -> None:
    text = _protocol()
    schema = _section(text, "Calibration decision schema", level=3)
    assert _csv_header(schema) == CALIBRATION_DECISION_HEADER
    rows = _table(schema, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in rows) == CALIBRATION_DECISION_HEADER

    gate = _normalized(_section(text, "Calibration and release gate", level=2))
    assert "decision MUST be revise" in gate
    assert "decision MUST be release" in gate
    assert (
        "A revise decision MUST NOT release any main-phase packet and MUST require a new "
        "protocol and coordinator snapshot." in gate
    )
    assert (
        "Reviewers from a passing calibration MAY continue only with previously unseen "
        "main-phase records." in gate
    )
    assert (
        "No reviewer exposed to failed-calibration ratings or discussion may rerate the "
        "stable 30 after revision." in gate
    )


def test_calibration_decision_snapshot_preimages_are_exact() -> None:
    text = _protocol()
    section = _section(
        text,
        "Calibration decision snapshot artifacts",
        level=3,
    )
    assert _csv_header(section) == CALIBRATION_DECISION_MANIFEST_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        CALIBRATION_DECISION_MANIFEST_HEADER
    )
    normalized = _normalized(section)
    assert (
        "The exact snapshot file set is decision.csv, candidate_ids.txt, "
        "assignment_ids.txt, manifest.csv, and SHA256SUMS; no other entry "
        "is allowed." in normalized
    )
    assert (
        "candidate_ids.txt MUST be exactly the 30 frozen coordinator "
        "calibration IDs in frozen sequence, one UTF-8 ID per LF-terminated "
        "line." in normalized
    )
    assert (
        "assignment_ids.txt MUST be exactly the 60 calibration assignment IDs "
        "sorted by UTF-8 bytes, one ID per LF-terminated line." in normalized
    )
    assert (
        "Validation MUST compare both preimage files byte-for-byte with the "
        "authoritative coordinator and calibration result snapshots." in normalized
    )


def test_reviewer_release_manifest_contract_is_exact() -> None:
    text = _protocol()
    section = _section(text, "Reviewer release manifest", level=3)
    assert _csv_header(section) == RELEASE_MANIFEST_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        RELEASE_MANIFEST_HEADER
    )
    normalized = _normalized(section)
    assert (
        "Every calibration and main reviewer release MUST contain canonical "
        "release_manifest.csv covered by SHA256SUMS." in normalized
    )
    assert (
        "For phase calibration, assignment_count MUST be 60 and both gate hash "
        "fields MUST be exactly NR." in normalized
    )
    assert (
        "For phase main, assignment_count MUST be 344 and the calibration "
        "result and calibration decision snapshot hashes MUST equal the coherent "
        "tuple that authorized publication." in normalized
    )
    assert (
        "The published release manifest MUST be validated after publication "
        "against its captured authorization inputs." in normalized
    )


def test_phase_result_manifest_binds_the_exact_reviewer_release() -> None:
    text = _protocol()
    section = _section(
        text,
        "Phase-result manifest and release binding",
        level=3,
    )
    assert _csv_header(section) == PHASE_RESULT_MANIFEST_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        PHASE_RESULT_MANIFEST_HEADER
    )
    normalized = _normalized(section)
    assert (
        "Calibration results MUST bind the exact calibration reviewer release."
        in normalized
    )
    assert (
        "Main results MUST bind the exact gated main reviewer release and MUST "
        "additionally be validated against the same calibration reviewer release, "
        "calibration result, and passing calibration decision" in normalized
    )
    assert (
        "result assignment set exactly with its released packet assignment set"
        in normalized
    )


def test_adjudication_is_independent_triggered_and_append_only() -> None:
    text = _protocol()
    section = _section(text, "Adjudication", level=2)
    triggers = _table(
        _section(text, "Adjudication triggers", level=3),
        ("Trigger", "Required condition"),
    )
    assert [row["Trigger"] for row in triggers] == ["A1", "A2", "A3", "A4"]
    conditions = " ".join(row["Required condition"] for row in triggers)
    assert "screening-status disagreement" in conditions
    assert "criterion disagreement" in conditions
    assert "unequal normalized exclusion reasons" in conditions
    assert "known unresolved screening conflict" in conditions
    assert "even when both ratings agree" in conditions

    normalized = _normalized(section)
    assert "The adjudicator MUST be a third reviewer distinct from both original reviewers." in normalized
    assert "Both locked raw ratings MUST be preserved byte-for-byte." in normalized
    assert "Adjudication MUST append a separate decision and MUST NOT rewrite either rating." in normalized


def test_adjudication_schema_and_a3_normalization_are_exact() -> None:
    text = _protocol()
    section = _section(text, "Adjudication", level=2)
    assert _csv_header(section) == ADJUDICATION_HEADER
    fields = _table(
        _section(text, "Adjudication field contract", level=3),
        ("Field", "Normative rule"),
    )
    assert tuple(_unquote(row["Field"]) for row in fields) == ADJUDICATION_HEADER

    normalized = _normalized(section)
    assert "Unicode NFKC" in normalized
    assert "casefold" in normalized
    assert (
        "replace every maximal run of nonalphanumeric characters with one ASCII space"
        in normalized
    )
    assert "collapse whitespace" in normalized
    assert "A3 is triggered if and only if the two normalized strings are unequal." in normalized


def test_reliability_reporting_uses_locked_pre_adjudication_ratings() -> None:
    text = _protocol()
    section = _section(text, "Reliability reporting", level=2)
    rows = _table(section, ("Metric", "Required calculation"))
    assert [row["Metric"] for row in rows] == [
        "Overall exact agreement",
        "Overall exact criterion agreement",
        "Category-specific agreement",
        "Nominal Krippendorff alpha",
        "Nominal Gwet AC1",
    ]
    calculations = {
        row["Metric"]: row["Required calculation"] for row in rows
    }
    assert calculations["Category-specific agreement"] == (
        "For each category `k`, define directional `a_k=n_kk`, "
        "`b_k=sum_(l!=k) n_kl`, `c_k=sum_(l!=k) n_lk`, and "
        "`d_k=N-a_k-b_k-c_k`; report `2a_k/(2a_k+b_k+c_k)` and "
        "`2d_k/(2d_k+b_k+c_k)`. A zero denominator is "
        "`not_estimable`."
    )
    assert calculations["Nominal Krippendorff alpha"] == (
        "Let `P_o=sum_k n_kk/N`, `D_o=1-P_o`, and pooled "
        "`n_k=sum_l(n_kl+n_lk)` over `2N` ratings. Use "
        "`D_e=sum_k n_k(2N-n_k)/(2N(2N-1))` and "
        "`alpha=1-D_o/D_e`; `D_e=0` is `not_estimable`."
    )
    assert calculations["Nominal Gwet AC1"] == (
        "With `K=2` and `p_k=n_k/(2N)`, use "
        "`P_e=sum_k p_k(1-p_k)/(K-1)` and "
        "`AC1=(P_o-P_e)/(1-P_e)`; `1-P_e=0` is "
        "`not_estimable`."
    )
    normalized = _normalized(section)
    assert (
        "For statuses k and l, n_kl is the number of candidates whose first "
        "assignment-ordered rating is k and whose second assignment-ordered "
        "rating is l." in normalized
    )
    assert (
        "All reliability statistics MUST use the two locked, pre-adjudication, final-version "
        "ratings for all 202 records." in normalized
    )
    assert "Adjudicated decisions MUST NOT replace raw ratings in reliability calculations." in normalized
    assert (
        "Overall exact criterion agreement MUST be reported as an exact numerator and "
        "denominator over all 202 candidates." in normalized
    )
    assert (
        "Status agreement, exact criterion agreement, nominal Krippendorff alpha, and "
        "nominal Gwet AC1 MUST each have a deterministic 95% candidate-bootstrap "
        "interval" in normalized
    )
    assert "Use exactly 10000 replicates per scope." in normalized
    assert "Each scope starts at (r=0,j=0)" in normalized
    assert "screening-bootstrap-v1" in normalized
    assert "combined-primary digest" in normalized
    assert "samples N paired candidate units with replacement" in normalized
    assert "valid-replicate count MUST be reported separately for each metric" in normalized


def test_automation_disclosure_independence_and_author_accountability() -> None:
    text = _protocol()
    section = _normalized(
        _section(text, "Automation, AI assistance, and accountability", level=2)
    )
    assert (
        "The paper MUST disclose whether automation or AI assistance was used and, if it "
        "was, identify its role, tool or model and version, instructions, affected stages, "
        "and verification procedure." in section
    )
    assert "The protocol MUST NOT represent an automated agent as a human reviewer." in section
    assert (
        "No conversation history, memory, ratings, results, or reviewer-produced "
        "retrieval state may be supplied across roles." in section
    )
    assert "they MUST NOT claim technical cache isolation." in section
    assert (
        "The immutable execution register MUST record the exact model identifier, model "
        "version, configuration hash, and prompt hash for every automated role." in section
    )
    assert (
        "The authors remain accountable for the final eligibility decisions, extracted "
        "evidence, analyses, and claims." in section
    )
    assert (
        "Before publication, accountable authors MUST verify all 202 final eligibility "
        "decisions and every deciding evidence locator against the cited source." in section
    )
    assert (
        "This codebook does not name any person or tool as having completed screening "
        "and does not assert that any planned action has occurred." in section
    )


def test_execution_register_schema_and_field_rules_are_exact() -> None:
    text = _protocol()
    section = _section(
        text,
        "Execution register schema",
        level=3,
    )
    assert _csv_header(section) == EXECUTION_REGISTER_HEADER
    rows = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in rows) == (
        EXECUTION_REGISTER_HEADER
    )
    normalized = _normalized(section)
    assert (
        "one row for each of the 404 screening assignments and one row for "
        "each required adjudication candidate" in normalized
    )
    assert "human, automated, or hybrid" in normalized
    assert "Canonical JSON object" in normalized
    assert "result_file_sha256" in normalized
    assert (
        "Exactly Fresh context; no shared conversation history, memory, ratings, "
        "results, or retrieval cache." in normalized
    )
    assert "reviewer execution contexts MUST be distinct" in normalized
    assert "adjudicator context MUST differ from both reviewer contexts" in normalized


def test_limited_provider_provenance_contract_is_exact() -> None:
    section = _section(
        _protocol(),
        "Execution register schema",
        level=3,
    )
    normalized = _normalized(section)
    assert FULL_CACHE_ISOLATION_STATEMENT in section
    assert LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT in section
    assert (
        "configuration_sha256, prompt_sha256, and user_instruction_sha256 "
        "remain REQUIRED lowercase 64-hex digests in both full-provider and "
        "limited-provider modes." in normalized
    )
    assert (
        "prompt_sha256 is the SHA-256 of the exact UTF-8 bytes of the rendered "
        "visible reviewer prompt; hidden system and developer instructions are "
        "excluded and have their own fields." in normalized
    )
    assert (
        "system_instruction_sha256 value of NR is permitted only when "
        "provider_metadata_limitations contains system_instruction_bytes with "
        "exact value provider-not-exposed." in normalized
    )
    assert (
        "developer_instruction_sha256 value of NR is permitted only when "
        "provider_metadata_limitations contains developer_instruction_bytes "
        "with exact value provider-not-exposed." in normalized
    )
    assert (
        "decoding_parameters MAY be NR only when provider_metadata_limitations "
        "contains decoding_parameters with exact value provider-not-exposed."
        in normalized
    )
    assert (
        "model_version MUST use the exact requested:<alias-or-date> form if and "
        "only if provider_metadata_limitations contains backend_model_version "
        "with exact value provider-not-exposed." in normalized
    )
    assert (
        "The limitations object MUST be a nonempty JSON object containing only "
        "the five documented keys and exact provider-not-exposed values."
        in normalized
    )
    assert (
        "Unknown keys, missing justifications, mismatched fields, unjustified "
        "limitations, and NR values without their corresponding limitation MUST "
        "be rejected." in normalized
    )
    assert (
        "Limited-provider mode MUST be configured and validated before any "
        "screening freeze." in normalized
    )
    example = json.dumps(
        {
            key: "provider-not-exposed"
            for key in PROVIDER_METADATA_LIMITATION_KEYS
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert example in section


def test_reviewer_prompt_template_is_exact_and_auditable() -> None:
    prompt = REVIEWER_PROMPT_PATH.read_text(encoding="utf-8")
    assert prompt.endswith("\n")
    for binding in (
        "ROLE_ID",
        "PROTOCOL_PATH",
        "PROTOCOL_SHA256",
        "PACKET_PATH",
        "PACKET_SHA256",
        "OUTPUT_PATH",
    ):
        assert f"{binding}: {{{{{binding}}}}}" in prompt

    csv_blocks = re.findall(r"```csv\n([^`]+)\n```", prompt)
    assert csv_blocks == [",".join(RESULT_HEADER)]
    assert "The protocol and assigned packet are the sole supplied screening inputs." in prompt
    assert (
        "No other conversation history, memory, ratings, results, summaries, or "
        "context may be supplied." in prompt
    )
    assert "Both duplicate reviewers MUST rate the same frozen evidence packet." in prompt
    assert (
        "Public retrieval during rating MAY verify metadata or report a packet defect "
        "but MUST NOT silently replace or add eligibility evidence." in prompt
    )
    assert "Stronger evidence after freeze requires a new packet version." in prompt
    assert "Write only canonical UTF-8 CSV to `OUTPUT_PATH`." in prompt
    assert "Do not emit a prose rating summary." in prompt
    assert (
        "`prompt_sha256` is the SHA-256 of the exact UTF-8 bytes of this rendered "
        "visible reviewer prompt." in prompt
    )
    assert (
        "```text\n"
        "ROLE_ID={{ROLE_ID}}\n"
        "ROWS_WRITTEN={{ROWS_WRITTEN}}\n"
        "OUTPUT_PATH={{OUTPUT_PATH}}\n"
        "OUTPUT_SHA256={{OUTPUT_SHA256}}\n"
        "```" in prompt
    )


def test_readme_automation_workflow_is_operationally_honest() -> None:
    readme = " ".join(README_PATH.read_text(encoding="utf-8").split())
    assert "Automated reviewer rows MUST use `human_role=NR`." in readme
    assert (
        "Each automated reviewer MUST start with `fork_context=false` in a fresh "
        "context." in readme
    )
    assert (
        "On a shared host, procedural isolation uses a separately generated random, "
        "role-private working and output path for each execution; this reduces accidental "
        "cross-role access but is not a claim of ACL, container, mount, or same-user "
        "process isolation." in readme
    )
    assert (
        "Provider retrieval-cache isolation is explicitly recorded as not exposed and "
        "reported as a residual limitation." in readme
    )
    assert (
        "Accountable-author verification of all 202 final decisions and every deciding "
        "evidence locator remains mandatory before publication." in readme
    )


def test_structured_adjudication_rationale_json_schema_is_exact() -> None:
    text = _protocol()
    raw_section = _section(
        text,
        "Structured adjudication rationale",
        level=3,
    )
    blocks = re.findall(r"```json\n([^\n]+)\n```", raw_section)
    assert len(blocks) == 1, "expected one canonical one-line JSON example"
    serialized = blocks[0]
    schema = json.loads(serialized)

    assert serialized == json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert tuple(schema) == (
        "comparison_analysis",
        "controlling_rules",
        "deciding_fact",
        "deciding_locator",
        "final_decision",
        "raw_exclusion_reasons",
        "raw_ratings",
        "resolved_conflicts",
        "schema_version",
        "source_url",
    )
    assert schema["schema_version"] == "1"
    assert tuple(schema["deciding_fact"]) == ("kind", "text")
    assert tuple(schema["final_decision"]) == (
        "criterion",
        "screening_status",
    )
    assert len(schema["raw_ratings"]) == 2
    assert all(
        tuple(rating) == ("assignment_id", "criterion", "screening_status")
        for rating in schema["raw_ratings"]
    )
    assert len(schema["raw_exclusion_reasons"]) == 2
    assert all(
        tuple(reason) == ("assignment_id", "reason")
        for reason in schema["raw_exclusion_reasons"]
    )
    assert len(schema["resolved_conflicts"]) == 1
    assert tuple(schema["resolved_conflicts"][0]) == (
        "conflict_id",
        "field",
        "value_a",
        "value_b",
    )

    section = _normalized(raw_section)
    assert "exactly these ten top-level keys and no others" in section
    assert "JSON string 1" in section
    assert "UTF-8" in section
    assert "ensure_ascii=False" in section
    assert "sort_keys=True" in section
    assert "separators=(\",\", \":\")" in section
    assert "allow_nan=False" in section
    assert "NaN, Infinity, and -Infinity are invalid" in section


def test_structured_adjudication_rationale_bindings_and_semantic_gate_are_exact() -> None:
    text = _protocol()
    section = _normalized(
        _section(text, "Structured adjudication rationale", level=3)
    )
    assert "raw_ratings MUST exactly equal the two locked ratings" in section
    assert (
        "controlling_rules MUST exactly equal the triggered A1 through A4 IDs"
        in section
    )
    assert "raw_exclusion_reasons MUST be an empty array unless A3 applies" in section
    assert (
        "resolved_conflicts MUST exactly equal the conflicts named by "
        "resolved_conflict_ids" in section
    )
    assert (
        "final_decision MUST exactly equal the adjudication row's screening_status "
        "and criterion" in section
    )
    assert "deciding_locator MUST exactly equal screening_locator" in section
    assert "source_url MUST equal one complete canonical URL in source_urls" in section
    assert "kind=exclusion_reason" in section
    assert "kind=retention_source_fact" in section
    assert "at least 120 characters" in section
    assert "18 alphabetic words" in section
    assert "12 distinct casefolded words" in section
    assert "at least ten alphabetic words and eight distinct words" in section
    assert "Prefix, suffix, or substring URL matches are invalid." in section
    assert "whereas" in section
    assert "MUST NOT be represented as proving semantic adequacy" in section
    assert "structurally validated but semantically pending" in section
    assert (
        "accountable-author verification of all 202 decisions and locators is the "
        "mandatory publication gate" in section
    )

def test_phase_result_validation_requires_an_authoritative_coordinator_anchor() -> None:
    text = _protocol()
    section = _normalized(
        _section(text, "Authoritative phase-result validation", level=3)
    )
    assert (
        "Public phase-result validation MUST always receive exactly one "
        "authoritative coordinator anchor" in section
    )
    assert (
        "coordinator snapshot path or an already captured coordinator snapshot"
        in section
    )
    assert (
        "Self-declared coordinator and protocol hashes in a phase-result manifest "
        "are integrity metadata only and MUST NOT be accepted as provenance." in section
    )


def test_a4_resolves_the_complete_authoritative_unresolved_conflict_set() -> None:
    text = _protocol()
    section = _normalized(_section(text, "Adjudication", level=2))
    assert (
        "A4 applies to the complete authoritative set of unresolved candidate "
        "screening_status conflicts in the frozen coordinator snapshot." in section
    )
    assert (
        "resolved_conflict_ids MUST exactly equal that complete set in UTF-8 byte "
        "order" in section
    )
    assert (
        "resolved_conflicts MUST reproduce every and only those authoritative conflicts"
        in section
    )


def test_adjudication_snapshot_binds_the_passing_calibration_decision() -> None:
    text = _protocol()
    section = _section(text, "Adjudication snapshot artifacts", level=2)
    assert _csv_header(section) == ADJUDICATION_MANIFEST_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        ADJUDICATION_MANIFEST_HEADER
    )
    normalized = _normalized(section)
    assert (
        "calibration_decision_snapshot_sha256 MUST identify the immutable "
        "calibration decision whose decision is release" in normalized
    )
    assert (
        "The exact snapshot file set is adjudications.csv, execution_registry.csv, "
        "manifest.csv, and SHA256SUMS; no other entry is allowed." in normalized
    )
    assert (
        "Adjudication MUST NOT be sealed or validated from a revise decision"
        in normalized
    )


def test_execution_register_binds_distinct_human_identities() -> None:
    text = _protocol()
    section = _normalized(
        _section(text, "Execution register schema", level=3)
    )
    assert (
        "human_role is a stable human identity/role identifier that identifies "
        "the accountable individual" in section
    )
    assert (
        "the paired human or hybrid reviewer human_role values MUST be distinct"
        in section
    )
    assert (
        "a human or hybrid adjudicator's human_role MUST differ from both paired "
        "human or hybrid reviewer human_role values" in section
    )


def test_citation_key_activation_uses_one_audited_append_only_ledger() -> None:
    text = _protocol()
    section = _section(text, "Citation-key activation ledger", level=2)
    assert _csv_header(section) == CITATION_KEY_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == CITATION_KEY_HEADER
    normalized = _normalized(section)
    assert (
        "A previously keyless included candidate MAY be activated only "
        "through this audited full citation_keys.csv ledger." in normalized
    )
    assert (
        "The ledger MUST preserve the coordinator citation_keys.csv rows as an exact "
        "append-only prefix." in normalized
    )
    assert (
        "New assignments MUST apply only to previously keyless candidates and MUST be "
        "appended in UTF-8 candidate_id byte order." in normalized
    )
    assert (
        "The projection MUST copy the complete canonical ledger to citation_keys.csv"
        in normalized
    )
    assert "citation_key_ledger_sha256" in normalized
    assert "citation_keys_sha256" in normalized


def test_author_verification_schema_is_exact_and_covers_all_candidates() -> None:
    text = _protocol()
    section = _section(text, "Accountable-author verification", level=2)
    assert _csv_header(section) == AUTHOR_VERIFICATION_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        AUTHOR_VERIFICATION_HEADER
    )
    normalized = _normalized(section)
    assert (
        "exactly one row for each of the 202 candidates in UTF-8 candidate_id byte "
        "order" in normalized
    )
    assert "decision_sha256" in normalized
    assert "evidence_versions_sha256" in normalized
    assert "deciding_locators_sha256" in normalized
    assert "[A-Za-z0-9][A-Za-z0-9._:-]{2,127}" in normalized
    assert "verified_role MUST be exactly accountable-author" in normalized
    assert "verification_status MUST be exactly verified" in normalized
    assert (
        "verification_evidence MUST be a substantive candidate-specific sign-off"
        in normalized
    )


def test_projection_snapshot_schema_and_publication_bindings_are_exact() -> None:
    text = _protocol()
    section = _section(text, "Screening projection snapshot", level=2)
    assert _csv_header(section) == PROJECTION_MANIFEST_HEADER
    fields = _table(section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == (
        PROJECTION_MANIFEST_HEADER
    )
    normalized = _normalized(section)
    assert (
        "The exact projection file set is candidates.csv, citation_keys.csv, "
        "conflicts.csv, screening_decisions.csv, screening_agreement.csv, "
        "author_verification.csv, manifest.csv, and SHA256SUMS; no other entry "
        "is allowed." in normalized
    )
    assert (
        "calibration_decision_snapshot_sha256 MUST identify the same immutable "
        "passing release decision bound by adjudication" in normalized
    )
    assert "citation_key_ledger_sha256" in normalized
    assert "author_verification_sha256" in normalized
    assert "citation_keys_sha256" in normalized
    assert (
        "Projection sealing is forbidden until all 202 accountable-author "
        "verification rows validate." in normalized
    )



def test_v7_root_and_working_contracts_are_byte_identical() -> None:
    assert PROTOCOL_PATH.read_bytes() == V7_PROTOCOL_PATH.read_bytes()
    assert REVIEWER_PROMPT_PATH.read_bytes() == V7_REVIEWER_PROMPT_PATH.read_bytes()


def test_v7_boundary_is_historical_only_and_forbidden_in_results() -> None:
    text = _protocol()
    statuses = _table(
        _section(text, "Screening statuses", level=3),
        ("Value", "Normative meaning"),
    )
    assert [_unquote(row["Value"]) for row in statuses] == ["included", "excluded"]
    normalized = _normalized(_section(text, "Screening statuses", level=3))
    assert "boundary is historical terminology only" in normalized
    assert "MUST NOT be assigned as a v7 result, criterion, or CSV value" in normalized


def test_supporting_transfer_actor_packet_property_and_mapping_are_explicit() -> None:
    section = _normalized(
        _section(_protocol(), "Eligibility and supporting-transfer clarification", level=3)
    )
    assert (
        "The survey protocol or accountable authors make the supporting transfer; "
        "the source need not claim that transfer." in section
    )
    assert (
        "Frozen packet evidence MUST directly establish at least one property in the "
        "closed supporting list: fixed-course requirement, interface, benchmark property, "
        "dataset property, metric, simulator constraint, or reporting practice." in section
    )
    assert "reviewer MUST record the concrete protocol-level mapping in notes" in section
    assert "Speculative future reuse is insufficient." in section
    assert "binary evidence-presence check, not Pass 2 ranking or coding" in section


def test_v7_preserves_controlled_field_access_and_provenance_contracts() -> None:
    text = _protocol()
    result_section = _section(text, "Reviewer result schema", level=2)
    fields = _table(result_section, ("Field", "Normative rule"))
    assert tuple(_unquote(row["Field"]) for row in fields) == RESULT_HEADER

    access_rows = _table(
        _section(text, "Access statuses", level=3),
        ("Value", "Normative meaning"),
    )
    assert {_unquote(row["Value"]) for row in access_rows} == ACCESS_STATUSES

    evidence = _normalized(_section(text, "Access and evidence rules", level=2))
    for field in (
        "source_urls",
        "evidence_version",
        "evidence_retrieved_on",
        "evidence_archive_url",
        "evidence_sha256",
        "screening_locator",
        "exclusion_reason",
        "notes",
    ):
        assert field in evidence
    assert "Only evidence_archive_url, evidence_sha256, exclusion_reason, notes, and resolved_conflict_ids may contain NR" in evidence


def test_v7_duplicate_review_uses_one_frozen_packet_without_silent_retrieval() -> None:
    section = _normalized(_section(_protocol(), "Independence and blinding", level=2))
    assert "Both duplicate reviewers MUST rate the same frozen evidence packet." in section
    assert (
        "Public retrieval during rating MAY verify metadata or report a packet defect but "
        "MUST NOT silently replace or add eligibility evidence." in section
    )
    assert "Stronger evidence after freeze requires a new packet version" in section
    assert "rerating" in section


def test_v7_pass_two_tiers_and_claim_guardrails_are_explicit() -> None:
    section = _normalized(
        _section(_protocol(), "Pass 2 evidence coding and claim limits", level=2)
    )
    assert "Pass 2 is multi-label" in section
    assert "survey_evidence_tier as core, supporting, or contextual" in section
    assert "Retained-source count is not method count." in section
    assert "Supporting evidence MUST NOT substantiate generation-method claims." in section
    assert "Contextual evidence MUST NOT support implementation or performance claims." in section


def test_duplicate_agreement_has_explicit_packet_audit_limitation() -> None:
    section = _normalized(_section(_protocol(), "Reliability reporting", level=2))
    assert (
        "Duplicate agreement measures consistency of interpretation of coordinator-curated "
        "frozen packets." in section
    )
    assert (
        "It does not independently estimate retrieval reliability, packet completeness, "
        "source authenticity, or evidence-selection bias." in section
    )
    assert "Packet assembly and audit are separate processes." in section
    assert "Packet defects require versioning and rerating." in section


def test_v7_calibration_is_fresh_stable_and_blind() -> None:
    section = _normalized(_section(_protocol(), "Calibration and release gate", level=2))
    assert "fresh stable-30 calibration" in section
    assert "six blind reviewer contexts" in section
    assert "60 valid ratings" in section
    assert "agreement >= 0.80" in section
    assert "no systematic ambiguity" in section
    assert "MUST NOT receive v3-v6 ratings or disagreements" in section
def test_evidence_packet_phase_release_contract_is_operational() -> None:
    root = Path("paper/data/screening_protocol.md")
    working = Path("paper/data/screening_work/v7/protocol.md")
    payload = root.read_bytes()
    text = payload.decode("utf-8")

    assert payload == working.read_bytes()
    assert "## Evidence packet phase releases" in text
    assert (
        "candidate_id,artifact_id,artifact_role,source_url,evidence_version,"
        "evidence_retrieved_on,access_status,evidence_archive_url,"
        "evidence_sha256,local_filename,redistribution_status,retrieval_notes"
    ) in text
    for label in (
        "doi_or_publisher",
        "title_author",
        "scholarly_index_or_repository",
        "official_page",
    ):
        assert label in text
    assert "Evidence binds to immutable reviewer phase releases" in text
    assert "concurrent hostile local archive writer" in text


def test_configuration_v3_staged_evidence_contract_is_operational() -> None:
    root = Path("paper/data/screening_protocol.md")
    working = Path("paper/data/screening_work/v7/protocol.md")
    payload = root.read_bytes()
    text = payload.decode("utf-8")

    assert payload == working.read_bytes()
    assert "`configuration_version` `3`" in text
    assert "allowed_screening_statuses" in text
    assert "allowed_inclusion_criteria" in text
    assert "evidence_packet_manifest_sha256" in text
    assert "evidence/<candidate_id>/<artifact_id>/<basename>" in text
    assert "metadata-only" in text
    assert "stage-local SHA-256" in text
    assert "procedural untracked artifacts" in text
