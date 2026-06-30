import csv
import hashlib
import subprocess
from pathlib import Path

import pytest

import paper.scripts.merge_candidates as merge_candidates_module

from paper.scripts.merge_candidates import (
    main as merge_candidates_main,
    merge_candidate_files,
)
ALIAS_HEADER = (
    "retired_candidate_id",
    "surviving_candidate_id",
    "reason",
    "evidence",
)


def alias_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(ALIAS_HEADER, "")
    row.update(updates)
    return row


def write_alias_rows(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...] = ALIAS_HEADER,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


CORRECTION_HEADER = (
    "candidate_id",
    "field",
    "old_value",
    "new_value",
    "reason",
    "evidence",
    "resolver",
)


def correction_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(CORRECTION_HEADER, "")
    row.update(updates)
    return row


def write_correction_rows(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...] = CORRECTION_HEADER,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


from paper.scripts.validate_corpus import HEADERS


def merge_candidate_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["candidates.csv"], "")
    row.update(
        candidate_id="agent-0001",
        title="A Candidate Track Generator",
        source_type="paper",
        discovery_stream="test-stream",
        discovery_query="test query",
        discovery_agent="pytest",
        screening_status="candidate",
        metadata_status="unverified",
    )
    row.update(updates)
    return row


def conflict_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["conflicts.csv"], "")
    row.update(updates)
    return row


def write_conflict_rows(
    path: Path, rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS["conflicts.csv"])
        writer.writeheader()
        writer.writerows(rows)


def read_conflict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def write_candidate_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS["candidates.csv"])
        writer.writeheader()
        writer.writerows(rows)


def build_merge_fixture(
    tmp_path: Path,
    existing_rows: list[dict[str, str]],
    *agent_groups: list[dict[str, str]],
) -> tuple[Path, list[Path]]:
    existing_path = tmp_path / "candidates.csv"
    write_candidate_rows(existing_path, existing_rows)
    agent_paths = []
    for index, rows in enumerate(agent_groups, start=1):
        path = tmp_path / f"agent-{index}.csv"
        write_candidate_rows(path, rows)
        agent_paths.append(path)
    return existing_path, agent_paths


def immutable_origin(
    path: Path,
    row: dict[str, str],
    label: str | None = None,
) -> str:
    digest_input = "\0".join(
        row[name] for name in HEADERS["candidates.csv"]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    source = label or path.resolve().as_posix()
    return f"{source}#{row['candidate_id']}@sha256:{digest}"


def test_merge_deduplicates_doi_prefixes_case_and_trailing_slash(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0042",
        title="Canonical Track Generation",
        doi="doi:10.1000/ABC",
        discovery_stream="bootstrap",
    )
    incoming = merge_candidate_row(
        candidate_id="BG-009",
        title="Canonical Track Generation",
        doi="https://doi.org/10.1000/abc/",
        discovery_stream="blind-ground",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0042"]
    assert conflicts == []


def test_merge_deduplicates_nfkd_casefolded_punctuation_free_titles(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0010",
        title="Stra\u00dfe: G\u00e9n\u00e9ration\u2014de   Pistes!",
        doi="",
    )
    incoming = merge_candidate_row(
        title="STRASSE generation de pistes",
        doi="NR",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == "C0010"
    assert conflicts == []


def test_merge_canonicalizes_all_discovery_provenance_with_semicolon_space(
    tmp_path,
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        doi="10.1000/provenance",
        discovery_stream="z-stream; m-stream",
        discovery_query="z query; m query",
        discovery_agent="z-agent; m-agent",
    )
    incoming = merge_candidate_row(
        doi="https://doi.org/10.1000/PROVENANCE",
        discovery_stream="a-stream; z-stream",
        discovery_query="a query; z query",
        discovery_agent="a-agent; z-agent",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["discovery_stream"] == "a-stream; m-stream; z-stream"
    assert merged[0]["discovery_query"] == "a query; m query; z query"
    assert merged[0]["discovery_agent"] == "a-agent; m-agent; z-agent"


@pytest.mark.parametrize(
    ("field", "current", "proposed"),
    [
        ("title", "Canonical Track", "A Different Track"),
        ("authors", "A. Author", "B. Author"),
        ("year", "2020", "2021"),
        ("venue", "Venue A", "Venue B"),
        ("doi", "10.1000/canonical", "10.1000/different"),
        ("url", "https://example.test/a", "https://example.test/b"),
        ("source_type", "paper", "software"),
    ],
)
def test_merge_preserves_existing_bibliography_and_records_conflicts(
    tmp_path, field, current, proposed
):
    existing = merge_candidate_row(
        candidate_id="C0008",
        title="Canonical Track",
        authors="A. Author",
        year="2020",
        venue="Venue A",
        doi="10.1000/canonical",
        url="https://example.test/a",
        source_type="paper",
    )
    incoming = dict(existing)
    incoming["candidate_id"] = "agent-conflict"
    incoming[field] = proposed
    existing[field] = current
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0][field] == current
    assert len(conflicts) == 1
    assert conflicts[0]["record_type"] == "candidate"
    assert conflicts[0]["record_key"] == "C0008"
    assert conflicts[0]["field"] == field
    assert {conflicts[0]["value_a"], conflicts[0]["value_b"]} == {
        current,
        proposed,
    }


def test_merge_deduplicates_repeated_equivalent_conflicts(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Canonical Track",
        authors="A. Author",
        doi="10.1000/repeated",
    )
    incoming_a = merge_candidate_row(
        candidate_id="agent-a",
        title="Canonical Track",
        authors="B. Author",
        doi="10.1000/repeated",
        discovery_agent="agent-a",
    )
    incoming_b = dict(incoming_a)
    incoming_b.update(candidate_id="agent-b", discovery_agent="agent-b")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming_a], [incoming_b]
    )

    _, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [(row["field"], row["value_b"]) for row in conflicts] == [
        ("authors", "B. Author")
    ]


def test_merge_treats_nr_bibliographic_cells_as_missing(tmp_path):
    incoming = merge_candidate_row(
        cite_key="AgentVerifiedKey",
        authors="NR",
        year=" nr ",
        venue="NR",
        doi="NR",
        url="NR",
        source_type="NR",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert conflicts == []
    assert merged[0]["title"] == "A Candidate Track Generator"
    for field in ("authors", "year", "venue", "doi", "url", "source_type"):
        assert merged[0][field] == ""


def test_multiple_nr_doi_rows_do_not_share_an_identity(tmp_path):
    first = merge_candidate_row(title="Alpha Track", doi="NR")
    second = merge_candidate_row(title="Beta Track", doi="nr")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [first, second]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0001", "C0002"]
    assert {row["title"] for row in merged} == {"Alpha Track", "Beta Track"}
    assert all(row["doi"] == "" for row in merged)
    assert conflicts == []


def test_generic_urls_do_not_deduplicate_different_titles(tmp_path):
    first = merge_candidate_row(
        title="Alpha Track",
        url="https://github.com/example/course-generator",
    )
    second = merge_candidate_row(
        title="Beta Track",
        url="https://github.com/example/course-generator",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [first, second]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 2


def test_arxiv_doi_and_url_bridge_title_variants(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0016",
        title="Original Preprint Title",
        doi="10.48550/arXiv.2401.01234v2",
        url="",
    )
    incoming = merge_candidate_row(
        title="Published Title Variant",
        doi="NR",
        url="https://arxiv.org/pdf/2401.01234v1.pdf?download=1",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == "C0016"
    assert [(row["field"], row["value_b"]) for row in conflicts] == [
        ("title", "Published Title Variant")
    ]


def test_merge_rejects_a_row_bridging_multiple_existing_identities(tmp_path):
    first = merge_candidate_row(
        candidate_id="C0001",
        title="Alpha Track",
        doi="10.1000/alpha",
    )
    second = merge_candidate_row(
        candidate_id="C0002",
        title="Beta Track",
        doi="10.1000/beta",
    )
    bridge = merge_candidate_row(
        title="Beta Track",
        doi="10.1000/alpha",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [first, second], [bridge]
    )

    with pytest.raises(
        ValueError,
        match=r"bridges multiple existing identities.*C0001.*C0002",
    ):
        merge_candidate_files(existing_path, agent_paths)


def test_new_candidates_cannot_inherit_agent_verification_or_cite_key(tmp_path):
    incoming = merge_candidate_row(
        cite_key="AgentClaimedVerifiedKey",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


def test_existing_bootstrap_verification_fields_remain_stable(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        cite_key="",
        doi="10.1000/bootstrap",
        metadata_status="unverified",
    )
    incoming = merge_candidate_row(
        cite_key="IncomingKey",
        doi="10.1000/bootstrap",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


def test_new_excluded_candidate_retains_specific_reason(tmp_path):
    incoming = merge_candidate_row(
        cite_key="ExcludedAgentKey",
        screening_status="excluded",
        exclusion_reason="Scenery only; no course geometry contribution.",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert (
        merged[0]["exclusion_reason"]
        == "Scenery only; no course geometry contribution."
    )
    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


@pytest.mark.parametrize("reason", ["", "NR", " nr "])
def test_new_excluded_candidate_requires_a_specific_reason(tmp_path, reason):
    incoming = merge_candidate_row(
        screening_status="excluded",
        exclusion_reason=reason,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    with pytest.raises(ValueError, match="excluded.*specific exclusion_reason"):
        merge_candidate_files(existing_path, agent_paths)


def test_next_stable_id_follows_maximum_without_filling_retired_gap(tmp_path):
    existing = [
        merge_candidate_row(candidate_id="C0071", title="Track 71"),
        merge_candidate_row(candidate_id="C0073", title="Track 73"),
        merge_candidate_row(candidate_id="C0100", title="Track 100"),
    ]
    incoming = merge_candidate_row(title="A Newly Discovered Track")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, existing, [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == [
        "C0071",
        "C0073",
        "C0100",
        "C0101",
    ]
    assert "C0072" not in {row["candidate_id"] for row in merged}


def test_merge_preserves_all_metadata_evidence_origins(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        doi="10.1000/evidence",
        metadata_evidence="source-b; source-a",
    )
    incoming = merge_candidate_row(
        doi="10.1000/evidence",
        metadata_evidence="source-a; source-c",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["metadata_evidence"] == "source-b; source-a; source-c"


def test_merge_csv_parse_errors_include_path_and_line(tmp_path):
    existing_path, _ = build_merge_fixture(tmp_path, [])
    bad_agent = tmp_path / "bad-agent.csv"
    bad_agent.write_text(
        ",".join(HEADERS["candidates.csv"])
        + '\n"agent-1,unterminated\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"bad-agent\.csv:\d+: CSV parse error: unexpected end of data",
    ):
        merge_candidate_files(existing_path, [bad_agent])


def test_merge_invalid_utf8_errors_include_path_and_cause(tmp_path):
    existing_path, _ = build_merge_fixture(tmp_path, [])
    bad_agent = tmp_path / "bad-utf8.csv"
    bad_agent.write_bytes(b"\xff")

    with pytest.raises(ValueError, match=r"bad-utf8\.csv: invalid UTF-8") as error:
        merge_candidate_files(existing_path, [bad_agent])

    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_merge_is_independent_of_agent_path_and_row_order(tmp_path):
    duplicate_a = merge_candidate_row(
        candidate_id="agent-a",
        title="Shared Track",
        authors="A. Author",
        discovery_agent="agent-a",
    )
    duplicate_b = merge_candidate_row(
        candidate_id="agent-b",
        title="Shared Track",
        authors="B. Author",
        discovery_agent="agent-b",
    )
    unique = merge_candidate_row(
        candidate_id="agent-c",
        title="Unique Track",
        discovery_agent="agent-c",
    )
    first_existing, first_agents = build_merge_fixture(
        tmp_path / "first",
        [],
        [duplicate_b, duplicate_a],
        [unique],
    )
    second_existing, second_agents = build_merge_fixture(
        tmp_path / "second",
        [],
        [duplicate_a, duplicate_b],
        [unique],
    )

    first_result = merge_candidate_files(first_existing, first_agents)
    second_result = merge_candidate_files(
        second_existing, list(reversed(second_agents))
    )

    assert first_result[0] == second_result[0]
    first_signatures = [
        {key: value for key, value in row.items() if key != "resolution_evidence"}
        for row in first_result[1]
    ]
    second_signatures = [
        {key: value for key, value in row.items() if key != "resolution_evidence"}
        for row in second_result[1]
    ]
    assert first_signatures == second_signatures


def test_rerunning_public_merge_preserves_semantics_and_origin_history(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/stable",
    )
    duplicate = merge_candidate_row(
        title="Stable Track",
        authors="Conflicting Author",
        doi="https://doi.org/10.1000/STABLE",
    )
    new = merge_candidate_row(title="New Track", doi="NR")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )

    first_merged, first_conflicts = merge_candidate_files(
        existing_path, agent_paths
    )
    write_candidate_rows(existing_path, first_merged)
    second_merged, second_conflicts = merge_candidate_files(
        existing_path, agent_paths
    )

    assert second_merged == first_merged
    assert [
        {key: value for key, value in row.items() if key != "resolution_evidence"}
        for row in second_conflicts
    ] == [
        {key: value for key, value in row.items() if key != "resolution_evidence"}
        for row in first_conflicts
    ]
    first_evidence = set(first_conflicts[0]["resolution_evidence"].split("; "))
    second_evidence = set(second_conflicts[0]["resolution_evidence"].split("; "))
    historical = f"value_b={immutable_origin(existing_path, existing)}"
    canonical = f"value_b={immutable_origin(existing_path, first_merged[0])}"
    assert historical in first_evidence
    assert canonical in first_evidence & second_evidence
    assert historical not in second_evidence
    assert all("@sha256:" in item for item in first_evidence | second_evidence)


def test_merge_cli_dry_run_reports_identity_stream_and_conflict_counts(
    tmp_path, capsys
):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        doi="10.1000/cli",
        discovery_stream="bootstrap",
    )
    duplicate = merge_candidate_row(
        title="Conflicting Published Title",
        doi="https://doi.org/10.1000/CLI",
        discovery_stream="blind-ground",
    )
    new = merge_candidate_row(
        title="New CLI Track",
        discovery_stream="aware-geometry-rl",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )
    original = existing_path.read_bytes()

    result = merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent",
            str(agent_paths[0]),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "mode=dry-run" in output
    assert "merged_total=2" in output
    assert "new_count=1" in output
    assert "duplicate_matches[doi][blind-ground]=1" in output
    assert "conflicts[title]=1" in output
    assert "conflict_total=1" in output
    assert existing_path.read_bytes() == original
    assert not (tmp_path / "conflicts.csv").exists()


def test_merge_cli_write_has_exact_schemas_and_is_idempotent(tmp_path, capsys):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/write",
    )
    duplicate = merge_candidate_row(
        title="Stable Track",
        authors="Conflicting Author",
        doi="10.1000/write",
    )
    new = merge_candidate_row(title="New Written Track")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent",
        str(agent_paths[0]),
        "--write",
    ]

    assert merge_candidates_main(arguments) == 0
    capsys.readouterr()
    conflicts_path = tmp_path / "conflicts.csv"
    first_candidates = existing_path.read_bytes()
    first_conflicts = conflicts_path.read_bytes()
    with existing_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS[
            "candidates.csv"
        ]
    with conflicts_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS[
            "conflicts.csv"
        ]

    assert merge_candidates_main(arguments) == 0
    capsys.readouterr()

    assert existing_path.read_bytes() == first_candidates
    assert conflicts_path.read_bytes() == first_conflicts


def test_explicit_exclusion_wins_for_a_new_identity_seen_as_candidate(tmp_path):
    candidate = merge_candidate_row(
        candidate_id="blind-1",
        title="Overlapping Discovery",
        doi="10.1000/overlap",
        screening_status="candidate",
        exclusion_reason="",
    )
    excluded = merge_candidate_row(
        candidate_id="aware-1",
        title="Overlapping Discovery",
        doi="10.1000/overlap",
        screening_status="excluded",
        exclusion_reason="No course geometry contribution.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [],
        [candidate],
        [excluded],
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "No course geometry contribution."
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [("screening_status", "candidate", "excluded")]
    assert merged[0]["metadata_status"] == "unverified"


@pytest.mark.parametrize(
    (
        "candidate_id",
        "existing_title",
        "incoming_title",
        "arxiv_id",
        "incoming_url",
    ),
    [
        (
            "C0016",
            "Replay-Guided Adversarial Environment Design / REPAIRED",
            "Replay-Guided Adversarial Environment Design",
            "2110.02439",
            (
                "https://arxiv.org/abs/2110.02439v3; "
                "https://github.com/facebookresearch/dcd"
            ),
        ),
        (
            "C0020",
            (
                "Emergent Complexity and Zero-shot Transfer via "
                "Unsupervised Environment Design / PAIRED"
            ),
            (
                "Emergent Complexity and Zero-shot Transfer via "
                "Unsupervised Environment Design"
            ),
            "2012.02096",
            (
                "https://github.com/facebookresearch/dcd; "
                "https://arxiv.org/pdf/2012.02096v2.pdf"
            ),
        ),
    ],
)
def test_semicolon_url_items_bridge_versioned_arxiv_abs_and_pdf_variants(
    tmp_path,
    candidate_id,
    existing_title,
    incoming_title,
    arxiv_id,
    incoming_url,
):
    existing = merge_candidate_row(
        candidate_id=candidate_id,
        title=existing_title,
        doi="",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        discovery_stream="bootstrap",
    )
    incoming = merge_candidate_row(
        candidate_id="aware-local",
        title=incoming_title,
        doi="NR",
        url=incoming_url,
        discovery_stream="aware-geometry-rl",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == candidate_id
    assert {row["field"] for row in conflicts} == {"title", "url"}


def test_pair_write_rolls_back_both_files_when_second_replace_fails(
    tmp_path, monkeypatch
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Existing Track",
        doi="10.1000/existing",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-new",
        title="New Track",
        doi="10.1000/new",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [])
    original_candidates = existing_path.read_bytes()
    original_conflicts = conflicts_path.read_bytes()
    real_replace = Path.replace
    injected = False

    def fail_second_data_replace(source: Path, target: Path):
        nonlocal injected
        if Path(target) == conflicts_path and not injected:
            injected = True
            raise OSError("injected second replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_data_replace)

    with pytest.raises(OSError, match="second replacement failure"):
        merge_candidates_main(
            [
                "--existing",
                str(existing_path),
                "--agent",
                str(agent_paths[0]),
                "--write",
            ]
        )

    assert injected
    assert existing_path.read_bytes() == original_candidates
    assert conflicts_path.read_bytes() == original_conflicts


def test_conflict_evidence_records_all_sources_and_marks_metadata_conflict(
    tmp_path,
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Canonical Track",
        authors="A. Author",
        doi="10.1000/context",
        metadata_status="unverified",
    )
    incoming_a = merge_candidate_row(
        candidate_id="agent-a",
        title="Canonical Track",
        authors="B. Author",
        doi="10.1000/context",
    )
    incoming_b = merge_candidate_row(
        candidate_id="agent-b",
        title="Canonical Track",
        authors="B. Author",
        doi="10.1000/context",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming_a], [incoming_b]
    )

    forward = merge_candidate_files(existing_path, agent_paths)
    reverse = merge_candidate_files(existing_path, list(reversed(agent_paths)))
    merged, conflicts = forward

    assert forward == reverse
    assert merged[0]["metadata_status"] == "conflict"
    assert len(conflicts) == 1
    assert conflicts[0]["resolution"] == ""
    assert conflicts[0]["resolver"] == ""
    assert set(conflicts[0]["resolution_evidence"].split("; ")) == {
        f"value_a={immutable_origin(existing_path, existing)}",
        f"value_a={immutable_origin(existing_path, merged[0])}",
        f"value_b={immutable_origin(agent_paths[0], incoming_a)}",
        f"value_b={immutable_origin(agent_paths[1], incoming_b)}",
    }


def test_excluded_status_propagates_to_existing_stable_candidate_deterministically(
    tmp_path,
):
    existing = merge_candidate_row(
        candidate_id="C0042",
        title="Stable Candidate",
        doi="10.1000/stable-exclusion",
        discovery_stream="bootstrap",
        screening_status="candidate",
        exclusion_reason="",
    )
    candidate = merge_candidate_row(
        candidate_id="blind-local",
        title="Stable Candidate",
        doi="10.1000/stable-exclusion",
        discovery_stream="blind-stream",
        screening_status="candidate",
    )
    excluded = merge_candidate_row(
        candidate_id="aware-local",
        title="Stable Candidate",
        doi="10.1000/stable-exclusion",
        discovery_stream="aware-stream",
        screening_status="excluded",
        exclusion_reason="No course geometry contribution.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [candidate], [excluded]
    )

    forward = merge_candidate_files(existing_path, agent_paths)
    reverse = merge_candidate_files(existing_path, list(reversed(agent_paths)))

    assert forward == reverse
    merged, conflicts = forward
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [("screening_status", "candidate", "excluded")]
    assert merged[0]["metadata_status"] == "unverified"
    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "No course geometry contribution."
    assert merged[0]["discovery_stream"] == (
        "aware-stream; blind-stream; bootstrap"
    )


def test_dry_run_reports_zero_inclusive_file_stream_and_identity_counts(
    tmp_path, capsys
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Existing Track",
        doi="10.1000/report",
    )
    duplicate = merge_candidate_row(
        candidate_id="agent-duplicate",
        title="Existing Track",
        doi="10.1000/report",
        discovery_stream="mixed-stream",
    )
    first_new = merge_candidate_row(
        candidate_id="agent-new-a",
        title="New Track A",
        discovery_stream="mixed-stream",
    )
    second_new = merge_candidate_row(
        candidate_id="agent-new-b",
        title="New Track B",
        discovery_stream="new-only-stream",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [existing],
        [duplicate, first_new],
        [second_new],
        [],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            *[
                item
                for path in agent_paths
                for item in ("--agent", str(path))
            ],
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "incoming_total=3" in output
    assert "source_file[agent-1.csv].incoming=2" in output
    assert "source_file[agent-1.csv].new=1" in output
    assert "source_file[agent-1.csv].duplicate=1" in output
    assert "source_file[agent-2.csv].incoming=1" in output
    assert "source_file[agent-2.csv].new=1" in output
    assert "source_file[agent-2.csv].duplicate=0" in output
    assert "source_file[agent-3.csv].incoming=0" in output
    assert "source_file[agent-3.csv].new=0" in output
    assert "source_file[agent-3.csv].duplicate=0" in output
    assert "source_stream[mixed-stream].incoming=2" in output
    assert "source_stream[mixed-stream].new=1" in output
    assert "source_stream[mixed-stream].duplicate=1" in output
    assert "source_stream[new-only-stream].incoming=1" in output
    assert "source_stream[new-only-stream].new=1" in output
    assert "source_stream[new-only-stream].duplicate=0" in output
    assert "identity_matches[doi]=1" in output
    assert "identity_matches[title]=0" in output
    assert "identity_matches[arxiv]=0" in output
    assert "conflicts[doi]=0" in output
    assert "conflicts[source_type]=0" in output


def test_identity_component_collapses_carracing_alias_chain_to_c0017(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0017",
                title="OpenAI Gym CarRacing environment",
                source_type="benchmark; simulator",
                url="https://github.com/Farama-Foundation/Gymnasium",
                discovery_stream="bootstrap",
                discovery_query="bootstrap",
                discovery_agent="bootstrap",
                metadata_status="verified",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="ASIM0001",
                title="Car Racing",
                source_type="documentation",
                url=(
                    "https://gymnasium.farama.org/environments/box2d/"
                    "car_racing/"
                ),
                metadata_evidence=(
                    "https://github.com/Farama-Foundation/Gymnasium/blob/"
                    "main/gymnasium/envs/box2d/car_racing.py"
                ),
                discovery_stream="agent-simulator",
                discovery_query="car racing docs",
                discovery_agent="sim-agent",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="AGRL0013",
                title="Gymnasium CarRacing track generator",
                source_type="official software",
                url=(
                    "https://github.com/Farama-Foundation/Gymnasium/blob/"
                    "main/gymnasium/envs/box2d/car_racing.py"
                ),
                discovery_stream="agent-rl",
                discovery_query="seed::C0017",
                discovery_agent="rl-agent",
            )
        ],
    )

    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0174",
                surviving_candidate_id="C0017",
                reason="Official documentation duplicates the system.",
                evidence=(
                    "https://gymnasium.farama.org/environments/box2d/"
                    "car_racing/"
                ),
            ),
            alias_row(
                retired_candidate_id="C0186",
                surviving_candidate_id="C0017",
                reason="Official generator duplicates the system.",
                evidence="https://github.com/Farama-Foundation/Gymnasium",
            ),
        ],
    )

    merged, _ = merge_candidate_files(
        existing_path, agent_paths, aliases_path=aliases_path
    )

    assert [row["candidate_id"] for row in merged] == ["C0017"]
    assert merged[0]["discovery_stream"] == (
        "agent-rl; agent-simulator; bootstrap"
    )


def test_pypi_project_versions_form_one_random_track_generator_component(
    tmp_path,
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [],
        [
            merge_candidate_row(
                candidate_id="BG0012",
                title="Random Track Generator",
                source_type="package",
                url=(
                    "https://pypi.org/project/"
                    "random-track-generator/1.0.0/"
                ),
            )
        ],
        [
            merge_candidate_row(
                candidate_id="ASIM0029",
                title="random-track-generator 1.1.0",
                source_type="software package",
                url=(
                    "https://pypi.org/project/"
                    "random_track_generator/1.1.0/"
                ),
            )
        ],
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0001"]


def test_gitlab_file_url_normalizes_to_nested_repository_root(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [],
        [
            merge_candidate_row(
                candidate_id="agent-root",
                title="Nested Track Repository",
                source_type="software repository",
                url="https://gitlab.com/group/subgroup/track-generator.git",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="agent-file",
                title="Nested Track Source",
                source_type="source artifact",
                metadata_evidence=(
                    "source=https://gitlab.com/group/subgroup/"
                    "track-generator/-/blob/main/generator.py"
                ),
            )
        ],
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0001"]
    assert [(row["field"], row["value_b"]) for row in conflicts] == [
        ("title", "Nested Track Source"),
        ("source_type", "source artifact"),
    ]

def test_seed_provenance_alone_does_not_deduplicate(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0015",
                title="Automated Lane Keeping Systems (ALKS)",
                source_type="standard",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="AGRL0018",
                title="A Distinct ALKS Research Paper",
                source_type="paper",
                discovery_query="seed::C0015",
            )
        ],
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0015", "C0016"]


def test_strong_arxiv_identity_wins_regardless_of_seed_lineage(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0014",
                title=(
                    "Automatic Generation of Challenging Road Networks for "
                    "ALKS Testing based on Bezier Curves and Search"
                ),
                url="https://arxiv.org/abs/2103.01288",
                source_type="paper",
            ),
            merge_candidate_row(
                candidate_id="C0015",
                title="Automated Lane Keeping Systems (ALKS)",
                source_type="standard",
            ),
        ],
        [
            merge_candidate_row(
                candidate_id="AGRL0018",
                title=(
                    "Automatic Generation of Challenging Road Networks for "
                    "ALKS Testing based on Bezier Curves and Search"
                ),
                doi="10.48550/arxiv.2103.01288",
                url="https://arxiv.org/abs/2103.01288",
                source_type="paper",
                discovery_query="seed::C0015",
            )
        ],
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0014", "C0015"]
    assert merged[0]["doi"] == "10.48550/arxiv.2103.01288"
    assert conflicts == []


def test_seed_target_must_exist_in_premerge_existing_ledger(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Baseline")],
        [
            merge_candidate_row(
                candidate_id="agent-new",
                title="Would Become C0002",
            ),
            merge_candidate_row(
                candidate_id="agent-seed",
                title="Invalid Seed",
                discovery_query="seed::C0002",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"seed target C0002 does not exist.*candidates\.csv",
    ):
        merge_candidate_files(existing_path, agent_paths)


def test_artifact_component_rejects_two_existing_stable_ids(tmp_path):
    shared_repository = "https://github.com/example/track-tool"
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="First Stable Tool",
                source_type="software",
                url=shared_repository,
            ),
            merge_candidate_row(
                candidate_id="C0002",
                title="Second Stable Tool",
                source_type="repository",
                url=f"{shared_repository}/blob/main/tool.py",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"baseline collision.*C0001.*C0002",
    ):
        merge_candidate_files(existing_path, [])


def test_transitive_title_repository_and_package_alias_chain_is_one_component(
    tmp_path,
):
    bridge = merge_candidate_row(
        candidate_id="agent-bridge",
        title="Stable Track Tool",
        source_type="software package",
        url="https://github.com/example/track-tool/tree/main",
        metadata_evidence="https://pypi.org/project/track-tool/2.0.0/",
    )
    package = merge_candidate_row(
        candidate_id="agent-package",
        title="Renamed Track Package",
        source_type="package",
        url="https://pypi.org/project/track_tool/",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0042",
                title="Stable Track Tool",
                source_type="software",
            )
        ],
        [bridge],
        [package],
    )

    forward = merge_candidate_files(existing_path, agent_paths)
    reverse = merge_candidate_files(existing_path, list(reversed(agent_paths)))

    assert forward == reverse
    assert [row["candidate_id"] for row in forward[0]] == ["C0042"]


@pytest.mark.parametrize(
    "paper_source_type",
    [
        "paper",
        "conference article; official software",
        "preprint; repository",
        "survey; package",
    ],
)
def test_metadata_artifact_aliases_never_merge_paper_like_rows(
    tmp_path, paper_source_type
):
    shared_source = "https://github.com/example/shared/blob/main/generator.py"
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [],
        [
            merge_candidate_row(
                candidate_id="paper-row",
                title="Research Paper",
                source_type=paper_source_type,
                metadata_evidence=shared_source,
            ),
            merge_candidate_row(
                candidate_id="artifact-row",
                title="Reusable Generator",
                source_type="software artifact",
                metadata_evidence=shared_source,
            ),
        ],
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 2


def test_normal_write_preserves_resolved_conflict_by_stable_signature(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/resolved",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-conflict",
        title="Stable Track",
        authors="Corrected Author",
        doi="10.1000/resolved",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    reviewed = conflict_row(
        conflict_id="REVIEWED-AUTHORS",
        record_type="candidate",
        record_key="C0001",
        field="authors",
        value_a="Original Author",
        value_b="Corrected Author",
        resolution="Use the registry form.",
        resolver="metadata-reviewer",
        resolution_evidence="registry snapshot 2026-06-30",
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [reviewed])

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0

    rows = read_conflict_rows(conflicts_path)
    assert len(rows) == 1
    result = rows[0]
    assert {
        name: result[name]
        for name in HEADERS["conflicts.csv"]
        if name != "resolution_evidence"
    } == {
        name: reviewed[name]
        for name in HEADERS["conflicts.csv"]
        if name != "resolution_evidence"
    }
    with existing_path.open(encoding="utf-8", newline="") as handle:
        canonical = next(csv.DictReader(handle))
    assert set(result["resolution_evidence"].split("; ")) == {
        "registry snapshot 2026-06-30",
        f"value_a={immutable_origin(existing_path, existing)}",
        f"value_a={immutable_origin(existing_path, canonical)}",
        f"value_b={immutable_origin(agent_paths[0], incoming)}",
    }


def test_new_conflict_id_is_hash_stable_when_earlier_conflict_is_added(
    tmp_path,
):
    def target_conflict_id(directory: Path, include_earlier: bool) -> str:
        target = merge_candidate_row(
            candidate_id="C0010",
            title="Target Track",
            authors="Original Target",
            doi="10.1000/target",
        )
        target_update = merge_candidate_row(
            candidate_id="target-agent",
            title="Target Track",
            authors="Updated Target",
            doi="10.1000/target",
        )
        existing = [target]
        incoming = [target_update]
        if include_earlier:
            existing.insert(
                0,
                merge_candidate_row(
                    candidate_id="C0001",
                    title="Earlier Track",
                    authors="Original Earlier",
                    doi="10.1000/earlier",
                ),
            )
            incoming.insert(
                0,
                merge_candidate_row(
                    candidate_id="earlier-agent",
                    title="Earlier Track",
                    authors="Updated Earlier",
                    doi="10.1000/earlier",
                ),
            )
        existing_path, agent_paths = build_merge_fixture(
            directory, existing, incoming
        )
        _, conflicts = merge_candidate_files(existing_path, agent_paths)
        return next(
            row["conflict_id"]
            for row in conflicts
            if row["record_key"] == "C0010"
        )

    alone = target_conflict_id(tmp_path / "alone", False)
    with_earlier = target_conflict_id(tmp_path / "with-earlier", True)

    assert alone == with_earlier
    assert alone.startswith("X")
    assert len(alone) == 13
    int(alone[1:], 16)


def test_component_conflict_evidence_tracks_true_value_origins_and_rows(
    tmp_path,
):
    first = merge_candidate_row(
        candidate_id="agent-a",
        title="Shared New Track",
        authors="A. Author",
        doi="10.1000/origin-chain",
    )
    second = merge_candidate_row(
        candidate_id="agent-b",
        title="Shared New Track",
        authors="B. Author",
        doi="10.1000/origin-chain",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [first], [second]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    author_conflict = next(row for row in conflicts if row["field"] == "authors")
    assert set(author_conflict["resolution_evidence"].split("; ")) == {
        f"value_a={immutable_origin(agent_paths[0], first)}",
        f"value_a={immutable_origin(existing_path, merged[0])}",
        f"value_b={immutable_origin(agent_paths[1], second)}",
    }


def test_replace_conflicts_explicitly_discards_unrelated_existing_rows(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/replace-conflicts",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-conflict",
        title="Stable Track",
        authors="Incoming Author",
        doi="10.1000/replace-conflicts",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(
        conflicts_path,
        [
            conflict_row(
                conflict_id="EVIDENCE-REVIEW",
                record_type="evidence",
                record_key="SomeEvidence",
                field="domain",
                value_a="ground",
                value_b="mixed",
                resolution="Keep both scopes.",
                resolver="reviewer",
                resolution_evidence="manual review",
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent",
            str(agent_paths[0]),
            "--replace-conflicts",
            "--write",
        ]
    ) == 0

    rows = read_conflict_rows(conflicts_path)
    assert len(rows) == 1
    assert rows[0]["record_type"] == "candidate"
    assert rows[0]["record_key"] == "C0001"
    assert rows[0]["field"] == "authors"


def test_normal_write_preserves_unrelated_candidate_and_evidence_conflicts(
    tmp_path,
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    preserved = [
        conflict_row(
            conflict_id="UNRELATED-CANDIDATE",
            record_type="candidate",
            record_key="C0001",
            field="venue",
            value_a="Venue A",
            value_b="Venue B",
            resolution="",
            resolver="",
            resolution_evidence="candidate review context",
        ),
        conflict_row(
            conflict_id="REVIEWED-EVIDENCE",
            record_type="evidence",
            record_key="SomeEvidence",
            field="domain",
            value_a="ground",
            value_b="mixed",
            resolution="Use mixed.",
            resolver="reviewer",
            resolution_evidence="scope review",
        ),
    ]
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, preserved)

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0

    assert read_conflict_rows(conflicts_path) == preserved


@pytest.mark.parametrize("reviewed_status", ["included", "boundary"])
def test_incoming_exclusion_cannot_overwrite_reviewed_status(
    tmp_path, reviewed_status
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Reviewed Track",
        doi="10.1000/reviewed-status",
        screening_status=reviewed_status,
        exclusion_reason="",
        metadata_status="verified",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-excluded",
        title="Reviewed Track",
        doi="10.1000/reviewed-status",
        screening_status="excluded",
        exclusion_reason="Agent exclusion rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == reviewed_status
    assert merged[0]["exclusion_reason"] == ""
    assert merged[0]["metadata_status"] == "verified"
    assert {
        row["field"]: (row["value_a"], row["value_b"])
        for row in conflicts
    } == {
        "screening_status": (reviewed_status, "excluded"),
        "exclusion_reason": ("<empty>", "Agent exclusion rationale."),
    }


def test_incoming_reason_cannot_overwrite_reviewed_exclusion(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Reviewed Exclusion",
        doi="10.1000/reviewed-exclusion",
        screening_status="excluded",
        exclusion_reason="Reviewed exclusion rationale.",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-excluded",
        title="Reviewed Exclusion",
        doi="10.1000/reviewed-exclusion",
        screening_status="excluded",
        exclusion_reason="Different agent rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "Reviewed exclusion rationale."
    assert len(conflicts) == 1
    assert conflicts[0]["field"] == "exclusion_reason"
    assert {conflicts[0]["value_a"], conflicts[0]["value_b"]} == {
        "Reviewed exclusion rationale.",
        "Different agent rationale.",
    }


def test_multiple_exclusion_reasons_remain_unreviewed_and_conflicted(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Unreviewed Track",
        doi="10.1000/ambiguous-exclusion",
        screening_status="candidate",
        exclusion_reason="",
    )
    first = merge_candidate_row(
        candidate_id="agent-first",
        title="Unreviewed Track",
        doi="10.1000/ambiguous-exclusion",
        screening_status="excluded",
        exclusion_reason="First specific rationale.",
    )
    second = merge_candidate_row(
        candidate_id="agent-second",
        title="Unreviewed Track",
        doi="10.1000/ambiguous-exclusion",
        screening_status="excluded",
        exclusion_reason="Second specific rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [first], [second]
    )

    forward = merge_candidate_files(existing_path, agent_paths)
    reverse = merge_candidate_files(existing_path, list(reversed(agent_paths)))
    merged, conflicts = forward

    assert forward == reverse
    assert merged[0]["screening_status"] == "candidate"
    assert merged[0]["exclusion_reason"] == ""
    assert merged[0]["metadata_status"] == "unverified"
    status_conflict = next(
        row for row in conflicts if row["field"] == "screening_status"
    )
    assert (status_conflict["value_a"], status_conflict["value_b"]) == (
        "candidate",
        "excluded",
    )
    reason_conflicts = [
        row for row in conflicts if row["field"] == "exclusion_reason"
    ]
    assert [
        (row["value_a"], row["value_b"]) for row in reason_conflicts
    ] == [
        ("<empty>", "First specific rationale."),
        ("<empty>", "Second specific rationale."),
    ]


def test_reviewed_exclusion_survives_candidate_rediscovery(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Reviewed Exclusion",
        doi="10.1000/excluded-rerun",
        screening_status="excluded",
        exclusion_reason="Reviewed exclusion rationale.",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-candidate",
        title="Reviewed Exclusion",
        doi="10.1000/excluded-rerun",
        screening_status="candidate",
        exclusion_reason="",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "Reviewed exclusion rationale."
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [("screening_status", "candidate", "excluded")]


def test_write_requires_at_least_one_agent_file(tmp_path, capsys):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
    )
    original = existing_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        merge_candidates_main(
            [
                "--existing",
                str(existing_path),
                "--write",
            ]
        )

    assert error.value.code == 2
    assert "--write requires at least one --agent-file" in capsys.readouterr().err
    assert existing_path.read_bytes() == original
    assert not (tmp_path / "conflicts.csv").exists()


def test_write_rejects_identical_candidate_and_conflict_targets(tmp_path):
    target = tmp_path / "conflicts.csv"
    write_candidate_rows(
        target,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
    )
    agent = tmp_path / "agent.csv"
    write_candidate_rows(
        agent,
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    original = target.read_bytes()

    with pytest.raises(
        ValueError,
        match="candidate and conflict output paths must be distinct",
    ):
        merge_candidates_main(
            [
                "--existing",
                str(target),
                "--agent-file",
                str(agent),
                "--write",
            ]
        )

    assert target.read_bytes() == original

def test_stage_failure_cleans_temporary_files_and_preserves_ledgers(
    tmp_path, monkeypatch
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [])
    original_candidates = existing_path.read_bytes()
    original_conflicts = conflicts_path.read_bytes()

    def fail_stage_fsync(_descriptor):
        raise OSError("injected stage fsync failure")

    monkeypatch.setattr(merge_candidates_module.os, "fsync", fail_stage_fsync)

    with pytest.raises(OSError, match="stage fsync failure"):
        merge_candidates_main(
            [
                "--existing",
                str(existing_path),
                "--agent-file",
                str(agent_paths[0]),
                "--write",
            ]
        )

    assert existing_path.read_bytes() == original_candidates
    assert conflicts_path.read_bytes() == original_conflicts
    assert list(tmp_path.glob(".*.tmp")) == []



def test_second_backup_failure_cleans_stages_and_first_backup(
    tmp_path, monkeypatch
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [])
    original_candidates = existing_path.read_bytes()
    original_conflicts = conflicts_path.read_bytes()
    real_copy2 = merge_candidates_module.shutil.copy2
    copy_count = 0

    def fail_second_backup(source, target, *args, **kwargs):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("injected second backup failure")
        return real_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(
        merge_candidates_module.shutil, "copy2", fail_second_backup
    )

    with pytest.raises(OSError, match="second backup failure"):
        merge_candidates_main(
            [
                "--existing",
                str(existing_path),
                "--agent-file",
                str(agent_paths[0]),
                "--write",
            ]
        )

    assert copy_count == 2
    assert existing_path.read_bytes() == original_candidates
    assert conflicts_path.read_bytes() == original_conflicts
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("metadata_status", ["unverified", "verified"])
def test_screening_only_conflict_does_not_change_metadata_status(
    tmp_path, metadata_status
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Screened Source",
        doi="10.1000/screening-only",
        screening_status="excluded",
        exclusion_reason="Reviewed rationale.",
        metadata_status=metadata_status,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-candidate",
        title="Screened Source",
        doi="10.1000/screening-only",
        screening_status="candidate",
        exclusion_reason="",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["field"] for row in conflicts] == ["screening_status"]
    assert merged[0]["metadata_status"] == metadata_status


def test_successful_pair_write_removes_stages_and_backups(tmp_path):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [])

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0

    assert list(tmp_path.glob(".*.tmp")) == []


def test_incomplete_rollback_retains_backups_and_reports_recovery_paths(
    tmp_path, monkeypatch
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Stable Track")],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [])
    original_candidates = existing_path.read_bytes()
    original_conflicts = conflicts_path.read_bytes()
    real_replace = Path.replace
    real_restore = merge_candidates_module._restore_file
    replacement_failed = False

    def fail_second_replacement(source: Path, target: Path):
        nonlocal replacement_failed
        if Path(target) == conflicts_path and not replacement_failed:
            replacement_failed = True
            raise OSError("injected second replacement failure")
        return real_replace(source, target)

    def fail_candidate_rollback(path: Path, backup: Path | None):
        if Path(path) == existing_path:
            raise OSError("injected candidate rollback failure")
        return real_restore(path, backup)

    monkeypatch.setattr(Path, "replace", fail_second_replacement)
    monkeypatch.setattr(
        merge_candidates_module, "_restore_file", fail_candidate_rollback
    )

    with pytest.raises(ValueError, match="rollback was incomplete") as error:
        merge_candidates_main(
            [
                "--existing",
                str(existing_path),
                "--agent-file",
                str(agent_paths[0]),
                "--write",
            ]
        )

    assert replacement_failed
    message = str(error.value)
    assert "recovery backups:" in message
    backups = sorted(tmp_path.glob(".*.backup.*.tmp"))
    assert len(backups) == 2
    assert all(str(path) in message for path in backups)
    candidate_backup = next(
        path for path in backups if path.name.startswith(".candidates.csv.")
    )
    conflict_backup = next(
        path for path in backups if path.name.startswith(".conflicts.csv.")
    )
    assert candidate_backup.read_bytes() == original_candidates
    assert conflict_backup.read_bytes() == original_conflicts


def test_automatic_exclusion_records_screening_disagreement_only(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Unreviewed Source",
        doi="10.1000/automatic-exclusion",
        screening_status="candidate",
        exclusion_reason="",
        metadata_status="unverified",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-excluded",
        title="Unreviewed Source",
        doi="10.1000/automatic-exclusion",
        screening_status="excluded",
        exclusion_reason="Specific exclusion rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "Specific exclusion rationale."
    assert merged[0]["metadata_status"] == "unverified"
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [("screening_status", "candidate", "excluded")]


def test_screening_conflict_write_rerun_is_byte_idempotent(tmp_path, capsys):
    candidate = merge_candidate_row(
        candidate_id="agent-candidate",
        title="Shared Screening Source",
        doi="10.1000/screening-rerun",
        screening_status="candidate",
        exclusion_reason="",
    )
    excluded = merge_candidate_row(
        candidate_id="agent-excluded",
        title="Shared Screening Source",
        doi="10.1000/screening-rerun",
        screening_status="excluded",
        exclusion_reason="Specific exclusion rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [candidate], [excluded]
    )
    common = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--agent-file",
        str(agent_paths[1]),
        "--write",
    ]

    assert merge_candidates_main([*common, "--replace-conflicts"]) == 0
    capsys.readouterr()
    conflicts_path = tmp_path / "conflicts.csv"
    first_candidates = existing_path.read_bytes()
    first_conflicts = conflicts_path.read_bytes()

    assert merge_candidates_main(common) == 0
    capsys.readouterr()

    assert existing_path.read_bytes() == first_candidates
    assert conflicts_path.read_bytes() == first_conflicts


def test_dry_run_reports_artifact_and_screening_conflict_counts(
    tmp_path, capsys
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Stable Artifact",
        source_type="software",
        url="https://github.com/example/track-tool",
        screening_status="candidate",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-artifact",
        title="Different Artifact Title",
        source_type="software",
        url="https://github.com/example/track-tool/blob/main/tool.py",
        discovery_query="seed::C0001",
        screening_status="excluded",
        exclusion_reason="Specific exclusion rationale.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "identity_matches[seed]" not in output
    assert "identity_matches[artifact]=1" in output
    assert "conflicts[screening_status]=1" in output
    assert "conflicts[exclusion_reason]=0" in output


def test_mirrored_paper_data_paths_use_portable_conflict_labels(tmp_path):
    data_dir = tmp_path / "staging" / "paper" / "data"
    existing_path = data_dir / "candidates.csv"
    agent_path = data_dir / "agent_runs" / "agent.csv"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/portable-path",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-row",
        title="Stable Track",
        authors="Incoming Author",
        doi="10.1000/portable-path",
    )
    write_candidate_rows(existing_path, [existing])
    write_candidate_rows(agent_path, [incoming])

    merged, conflicts = merge_candidate_files(existing_path, [agent_path])

    author_conflict = next(row for row in conflicts if row["field"] == "authors")
    agent_label = "paper/data/agent_runs/agent.csv"
    candidate_label = "paper/data/candidates.csv"
    assert set(author_conflict["resolution_evidence"].split("; ")) == {
        f"value_a={immutable_origin(agent_path, incoming, agent_label)}",
        f"value_b={immutable_origin(existing_path, existing, candidate_label)}",
        f"value_b={immutable_origin(existing_path, merged[0], candidate_label)}",
    }

def test_alias_migration_selects_survivor_and_preserves_unaffected_ids(
    tmp_path,
):
    survivor = merge_candidate_row(
        candidate_id="C0001",
        title="Canonical Track System",
        discovery_stream="bootstrap",
    )
    retired = merge_candidate_row(
        candidate_id="C0002",
        title="Legacy Track Placeholder",
        discovery_stream="legacy-catalog",
    )
    unaffected = merge_candidate_row(
        candidate_id="C0005",
        title="Unaffected Source",
        discovery_stream="bootstrap",
    )
    existing_path, _ = build_merge_fixture(
        tmp_path, [survivor, retired, unaffected]
    )
    aliases_path = tmp_path / "candidate_aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Legacy placeholder duplicates the citable system.",
                evidence="https://doi.org/10.1000/canonical",
            )
        ],
    )

    merged, conflicts = merge_candidate_files(
        existing_path, [], aliases_path=aliases_path
    )

    assert [row["candidate_id"] for row in merged] == ["C0001", "C0005"]
    assert merged[0]["discovery_stream"] == "bootstrap; legacy-catalog"
    assert merged[1] == unaffected
    assert {row["record_key"] for row in conflicts} == {"C0001"}


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (
            [
                alias_row(
                    retired_candidate_id="C1",
                    surviving_candidate_id="C0001",
                    reason="Invalid retired ID.",
                    evidence="https://example.test/evidence",
                )
            ],
            "retired_candidate_id.*not a stable candidate ID",
        ),
        (
            [
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0002",
                    reason="Self alias.",
                    evidence="https://example.test/evidence",
                )
            ],
            "cannot retire C0002 to itself",
        ),
        (
            [
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0001",
                    reason="First declaration.",
                    evidence="https://example.test/first",
                ),
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0001",
                    reason="Duplicate declaration.",
                    evidence="https://example.test/second",
                ),
            ],
            "duplicate retired candidate C0002",
        ),
        (
            [
                alias_row(
                    retired_candidate_id="C0003",
                    surviving_candidate_id="C0002",
                    reason="Indirect alias.",
                    evidence="https://example.test/indirect",
                ),
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0001",
                    reason="Direct alias.",
                    evidence="https://example.test/direct",
                ),
            ],
            "aliases must be direct and acyclic",
        ),
        (
            [
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0001",
                    reason="",
                    evidence="https://example.test/evidence",
                )
            ],
            "reason and evidence must be nonempty",
        ),
        (
            [
                alias_row(
                    retired_candidate_id="C0002",
                    surviving_candidate_id="C0099",
                    reason="Missing survivor.",
                    evidence="https://example.test/evidence",
                )
            ],
            "alias survivors are absent.*C0099",
        ),
    ],
)
def test_alias_validation_rejects_invalid_mappings(tmp_path, rows, match):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(candidate_id="C0001", title="One"),
            merge_candidate_row(candidate_id="C0002", title="Two"),
            merge_candidate_row(candidate_id="C0003", title="Three"),
        ],
    )
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(aliases_path, rows)

    with pytest.raises(ValueError, match=match):
        merge_candidate_files(existing_path, [], aliases_path=aliases_path)


def test_alias_validation_requires_exact_header(tmp_path):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="One")],
    )
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Legacy duplicate.",
                evidence="https://example.test/evidence",
            )
        ],
        header=(
            "surviving_candidate_id",
            "retired_candidate_id",
            "reason",
            "evidence",
        ),
    )

    with pytest.raises(ValueError, match="alias columns must be exactly"):
        merge_candidate_files(existing_path, [], aliases_path=aliases_path)


def test_alias_component_requires_every_non_survivor_to_be_retired(tmp_path):
    existing = [
        merge_candidate_row(
            candidate_id=candidate_id,
            title=f"Track {candidate_id}",
            doi="10.1000/shared-component",
        )
        for candidate_id in ("C0001", "C0002", "C0003")
    ]
    existing_path, _ = build_merge_fixture(tmp_path, existing)
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Declared duplicate.",
                evidence="https://doi.org/10.1000/shared-component",
            )
        ],
    )

    with pytest.raises(ValueError, match="complete retirement aliases"):
        merge_candidate_files(existing_path, [], aliases_path=aliases_path)


def test_absent_retired_alias_reserves_future_id_and_default_sibling_is_used(
    tmp_path,
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Survivor")],
        [merge_candidate_row(candidate_id="agent-new", title="New identity")],
    )
    write_alias_rows(
        tmp_path / "candidate_aliases.csv",
        [
            alias_row(
                retired_candidate_id="C0100",
                surviving_candidate_id="C0001",
                reason="Previously migrated duplicate.",
                evidence="https://example.test/migration",
            )
        ],
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0001", "C0101"]


def test_cli_accepts_explicit_alias_path(tmp_path, capsys):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(candidate_id="C0001", title="Survivor"),
            merge_candidate_row(candidate_id="C0002", title="Retired"),
        ],
    )
    aliases_path = tmp_path / "config" / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Explicitly retired placeholder.",
                evidence="https://example.test/migration",
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--aliases",
            str(aliases_path),
        ]
    ) == 0

    assert "merged_total=1" in capsys.readouterr().out


def test_candidate_excluded_conflict_is_symmetric_when_excluded_is_base(
    tmp_path,
):
    excluded = merge_candidate_row(
        candidate_id="agent-excluded",
        title="Symmetric Screening",
        doi="10.1000/symmetric-screening",
        discovery_stream="a-excluded",
        screening_status="excluded",
        exclusion_reason="No generated course geometry.",
    )
    candidate = merge_candidate_row(
        candidate_id="agent-candidate",
        title="Symmetric Screening",
        doi="10.1000/symmetric-screening",
        discovery_stream="z-candidate",
        screening_status="candidate",
        exclusion_reason="",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [excluded], [candidate]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "No generated course geometry."
    assert merged[0]["metadata_status"] == "unverified"
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [("screening_status", "candidate", "excluded")]


def test_all_excluded_component_keeps_deterministic_reason_and_conflicts_others(
    tmp_path,
):
    observations = [
        merge_candidate_row(
            candidate_id="agent-third",
            title="Unanimously Excluded",
            doi="10.1000/unanimous-exclusion",
            discovery_stream="third",
            screening_status="excluded",
            exclusion_reason="Third specific reason.",
        ),
        merge_candidate_row(
            candidate_id="agent-first",
            title="Unanimously Excluded",
            doi="10.1000/unanimous-exclusion",
            discovery_stream="first",
            screening_status="excluded",
            exclusion_reason="First specific reason.",
        ),
        merge_candidate_row(
            candidate_id="agent-second",
            title="Unanimously Excluded",
            doi="10.1000/unanimous-exclusion",
            discovery_stream="second",
            screening_status="excluded",
            exclusion_reason="Second specific reason.",
        ),
    ]
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], *([row] for row in observations)
    )

    forward = merge_candidate_files(existing_path, agent_paths)
    reverse = merge_candidate_files(existing_path, list(reversed(agent_paths)))
    merged, conflicts = forward

    assert forward == reverse
    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "First specific reason."
    assert merged[0]["metadata_status"] == "unverified"
    assert [row["field"] for row in conflicts] == [
        "exclusion_reason",
        "exclusion_reason",
    ]
    assert {
        frozenset((row["value_a"], row["value_b"])) for row in conflicts
    } == {
        frozenset(("First specific reason.", "Second specific reason.")),
        frozenset(("First specific reason.", "Third specific reason.")),
    }


def test_generated_conflict_pair_and_id_are_orientation_independent(tmp_path):
    def generated_conflict(
        directory: Path, existing_authors: str, incoming_authors: str
    ) -> dict[str, str]:
        existing_path, agent_paths = build_merge_fixture(
            directory,
            [
                merge_candidate_row(
                    candidate_id="C0001",
                    title="Orientation Track",
                    authors=existing_authors,
                    doi="10.1000/orientation",
                )
            ],
            [
                merge_candidate_row(
                    candidate_id="agent-orientation",
                    title="Orientation Track",
                    authors=incoming_authors,
                    doi="10.1000/orientation",
                )
            ],
        )
        _, conflicts = merge_candidate_files(existing_path, agent_paths)
        return next(row for row in conflicts if row["field"] == "authors")

    forward = generated_conflict(tmp_path / "forward", "Zulu", "Alpha")
    reverse = generated_conflict(tmp_path / "reverse", "Alpha", "Zulu")

    assert (forward["value_a"], forward["value_b"]) == ("Alpha", "Zulu")
    assert (reverse["value_a"], reverse["value_b"]) == ("Alpha", "Zulu")
    assert forward["conflict_id"] == reverse["conflict_id"]


def test_reversed_resolved_conflict_preserves_review_and_merges_all_origins(
    tmp_path,
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Reviewed Orientation",
                authors="Alpha Author",
                doi="10.1000/reviewed-orientation",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="agent-zulu-a",
                title="Reviewed Orientation",
                authors="Zulu Author",
                doi="10.1000/reviewed-orientation",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="agent-zulu-b",
                title="Reviewed Orientation",
                authors="Zulu Author",
                doi="10.1000/reviewed-orientation",
            )
        ],
    )
    reviewed = conflict_row(
        conflict_id="REVIEWED-ORIENTATION",
        record_type="candidate",
        record_key="C0001",
        field="authors",
        value_a="Zulu Author",
        value_b="Alpha Author",
        resolution="Use the authority record.",
        resolver="metadata-reviewer",
        resolution_evidence=(
            "manual registry review; value_a=legacy/source#C0001"
        ),
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(conflicts_path, [reviewed])

    arguments = ["--existing", str(existing_path)]
    for path in agent_paths:
        arguments.extend(("--agent-file", str(path)))
    arguments.append("--write")
    assert merge_candidates_main(arguments) == 0

    rows = read_conflict_rows(conflicts_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["conflict_id"] == "REVIEWED-ORIENTATION"
    assert (row["value_a"], row["value_b"]) == (
        "Zulu Author",
        "Alpha Author",
    )
    assert row["resolution"] == "Use the authority record."
    assert row["resolver"] == "metadata-reviewer"
    evidence = set(row["resolution_evidence"].split("; "))
    assert "manual registry review" in evidence
    assert "value_a=legacy/source#C0001" not in evidence
    origins = {
        item for item in evidence if item.startswith(("value_a=", "value_b="))
    }
    assert len(origins) == 4
    assert all("@sha256:" in item for item in origins)
    assert sum("#agent-zulu-" in item for item in origins) == 2
    assert sum("candidates.csv#C0001" in item for item in origins) == 2
    assert "@row:" not in row["resolution_evidence"]


def test_conflict_evidence_is_stable_when_agent_rows_are_reordered(tmp_path):
    existing_path = tmp_path / "candidates.csv"
    agent_path = tmp_path / "agent.csv"
    write_candidate_rows(existing_path, [])
    alpha = merge_candidate_row(
        candidate_id="agent-alpha",
        title="Row Stable Track",
        authors="Alpha Author",
        doi="10.1000/row-stability",
    )
    zulu = merge_candidate_row(
        candidate_id="agent-zulu",
        title="Row Stable Track",
        authors="Zulu Author",
        doi="10.1000/row-stability",
    )
    unrelated = merge_candidate_row(
        candidate_id="agent-unrelated",
        title="Unrelated Track",
    )
    write_candidate_rows(agent_path, [alpha, zulu, unrelated])

    first = merge_candidate_files(existing_path, [agent_path])
    write_candidate_rows(agent_path, [unrelated, zulu, alpha])
    second = merge_candidate_files(existing_path, [agent_path])

    assert first == second
    assert set(first[1][0]["resolution_evidence"].split("; ")) == {
        f"value_a={immutable_origin(agent_path, alpha)}",
        f"value_b={immutable_origin(agent_path, zulu)}",
        f"value_a={immutable_origin(existing_path, first[0][0])}",
    }
    assert "@row:" not in first[1][0]["resolution_evidence"]


def test_normal_write_migrates_retired_conflict_keys_to_survivor(
    tmp_path,
):
    shared = dict(
        title="Aliased Stable Track",
        authors="Stable Author",
        doi="10.1000/aliased-stable",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(candidate_id="C0001", **shared),
            merge_candidate_row(candidate_id="C0002", **shared),
        ],
        [merge_candidate_row(candidate_id="agent-new", title="New Track")],
    )
    write_alias_rows(
        tmp_path / "candidate_aliases.csv",
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Duplicate stable assignment.",
                evidence="https://doi.org/10.1000/aliased-stable",
            )
        ],
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(
        conflicts_path,
        [
            conflict_row(
                conflict_id="LEGACY-C0002",
                record_type="candidate",
                record_key="C0002",
                field="venue",
                value_a="Venue A",
                value_b="Venue B",
                resolution="Reviewed on the retired row.",
                resolver="reviewer",
                resolution_evidence="authority record",
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0

    migrated = next(
        row
        for row in read_conflict_rows(conflicts_path)
        if row["conflict_id"] == "LEGACY-C0002"
    )
    assert migrated["record_key"] == "C0001"
    assert migrated["resolution"] == "Reviewed on the retired row."


def test_report_enumerates_every_present_record_type_and_field(
    tmp_path, capsys
):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Reporting Track",
                authors="Original Author",
                doi="10.1000/report-types",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="agent-report",
                title="Reporting Track",
                authors="Incoming Author",
                doi="10.1000/report-types",
            )
        ],
    )
    write_conflict_rows(
        tmp_path / "conflicts.csv",
        [
            conflict_row(
                conflict_id="EVIDENCE-DOMAIN",
                record_type="evidence",
                record_key="E0001",
                field="domain",
                value_a="ground",
                value_b="mixed",
                resolution="Use mixed.",
                resolver="reviewer",
                resolution_evidence="manual scope review",
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "conflict_total=2" in output
    assert "conflicts[authors]=1" in output
    assert "conflicts_by_type[candidate][authors]=1" in output
    assert "conflicts_by_type[evidence][domain]=1" in output


def test_reconciliation_normalizes_legacy_row_number_origins(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Legacy Origin Track",
        authors="Alpha Author",
        doi="10.1000/legacy-origin",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-zulu",
        title="Legacy Origin Track",
        authors="Zulu Author",
        doi="10.1000/legacy-origin",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    conflicts_path = tmp_path / "conflicts.csv"
    write_conflict_rows(
        conflicts_path,
        [
            conflict_row(
                conflict_id="LEGACY-ORIGINS",
                record_type="candidate",
                record_key="C0001",
                field="authors",
                value_a="Alpha Author",
                value_b="Zulu Author",
                resolution="Use the authority form.",
                resolver="reviewer",
                resolution_evidence=(
                    "authority snapshot; "
                    f"value_a={existing_path.resolve().as_posix()}"
                    "#C0001@row:2; "
                    f"value_b={agent_paths[0].resolve().as_posix()}"
                    "#agent-zulu@row:99"
                ),
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
            "--write",
        ]
    ) == 0
    with existing_path.open(encoding="utf-8", newline="") as handle:
        canonical = next(csv.DictReader(handle))

    rows = read_conflict_rows(conflicts_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["conflict_id"] == "LEGACY-ORIGINS"
    assert row["resolution"] == "Use the authority form."
    assert row["resolver"] == "reviewer"
    evidence = set(row["resolution_evidence"].split("; "))
    assert evidence == {
        "authority snapshot",
        f"value_a={immutable_origin(existing_path, existing)}",
        f"value_a={immutable_origin(existing_path, canonical)}",
        f"value_b={immutable_origin(agent_paths[0], incoming)}",
    }
    assert "@row:" not in row["resolution_evidence"]
    for item in evidence:
        if item.startswith(("value_a=", "value_b=")):
            assert "@sha256:" in item


def test_committed_aliases_and_canonical_candidate_id_gaps():
    repository = Path(__file__).resolve().parents[1]
    aliases_path = repository / "paper" / "data" / "candidate_aliases.csv"
    candidates_path = repository / "paper" / "data" / "candidates.csv"
    conflicts_path = repository / "paper" / "data" / "conflicts.csv"

    with aliases_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == ALIAS_HEADER
        aliases = list(reader)
    expected_aliases = {
        "C0105": "C0003",
        "C0117": "C0097",
        "C0174": "C0017",
        "C0179": "C0021",
        "C0181": "C0180",
        "C0186": "C0017",
        "C0197": "C0196",
        "C0208": "C0147",
    }
    assert {
        row["retired_candidate_id"]: row["surviving_candidate_id"]
        for row in aliases
    } == expected_aliases
    assert all(row["reason"] and row["evidence"] for row in aliases)

    with candidates_path.open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in candidates}
    candidate_ids = set(by_id)
    retired_ids = set(expected_aliases)
    expected_gaps = {"C0072", *retired_ids}

    assert len(candidates) == 202
    assert max(int(candidate_id[1:]) for candidate_id in candidate_ids) == 211
    assert {
        f"C{number:04d}"
        for number in range(1, 212)
        if f"C{number:04d}" not in candidate_ids
    } == expected_gaps
    assert retired_ids.isdisjoint(candidate_ids)
    assert set(expected_aliases.values()) <= candidate_ids
    assert by_id["C0184"]["title"] == "Formula Student Driverless Simulator v2.2.0"
    assert by_id["C0185"]["title"] == "FSSIM: Formula Student Driverless Simulator"

    with conflicts_path.open(encoding="utf-8", newline="") as handle:
        conflicts = list(csv.DictReader(handle))
    assert not {
        row["record_key"]
        for row in conflicts
        if row["record_type"] == "candidate"
    } & retired_ids
    corrected_f1tenth_url = (
        "https://proceedings.mlr.press/v123/o-kelly20a.html"
    )
    assert by_id["C0180"]["url"] == corrected_f1tenth_url
    resolved_f1tenth = next(
        row
        for row in conflicts
        if row["record_type"] == "candidate"
        and row["record_key"] == "C0180"
        and row["field"] == "url"
        and "https://proceedings.mlr.press/v123/o2020a.html"
        in {row["value_a"], row["value_b"]}
        and corrected_f1tenth_url
        in {row["value_a"], row["value_b"]}
    )
    assert resolved_f1tenth["resolution"] == corrected_f1tenth_url
    assert resolved_f1tenth["resolver"] == "entity-resolution-audit"
    assert corrected_f1tenth_url in resolved_f1tenth["resolution_evidence"]
    assert all("@row:" not in row["resolution_evidence"] for row in conflicts)



def test_secondary_repository_evidence_does_not_bridge_distinct_primary_rows(
    tmp_path,
):
    guide = merge_candidate_row(
        candidate_id="C0207",
        title="Virtual RobotX Competition 2022 Technical Guide",
        url=(
            "https://robonation.org/app/uploads/VRX2022-Technical-Guide.pdf"
        ),
        source_type="official competition documentation; official repository",
        metadata_evidence="https://github.com/osrf/vrx",
    )
    resources = merge_candidate_row(
        candidate_id="C0208",
        title="Virtual RobotX Resources",
        url="https://github.com/osrf/vrx",
        source_type="official repository",
        metadata_evidence="https://github.com/osrf/vrx",
    )
    existing_path, _ = build_merge_fixture(tmp_path, [guide, resources])

    merged, conflicts = merge_candidate_files(existing_path, [])

    assert [row["candidate_id"] for row in merged] == ["C0207", "C0208"]
    assert conflicts == []


def test_absent_retired_alias_redirects_matching_primary_evidence_to_survivor(
    tmp_path,
):
    survivor = merge_candidate_row(
        candidate_id="C0001",
        title="System Publication",
        url="https://proceedings.example.test/system-paper",
        source_type="conference paper",
        discovery_stream="publication",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-repository",
        title="Official System Repository",
        url="https://github.com/example/system",
        source_type="official repository",
        discovery_stream="repository",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [survivor], [incoming]
    )
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Repository implements the stable system paper.",
                evidence=(
                    "https://github.com/example/system; "
                    "https://proceedings.example.test/system-paper"
                ),
            )
        ],
    )

    merged, conflicts = merge_candidate_files(
        existing_path, agent_paths, aliases_path=aliases_path
    )

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == "C0001"
    assert merged[0]["discovery_stream"] == "publication; repository"
    assert {row["record_key"] for row in conflicts} == {"C0001"}


def test_retirement_repository_evidence_does_not_bridge_shared_paper_repo(
    tmp_path,
):
    survivor = merge_candidate_row(
        candidate_id="C0001",
        title="Target System Paper",
        url="https://proceedings.example.test/target-paper",
        source_type="conference paper; official software",
    )
    unrelated = merge_candidate_row(
        candidate_id="C0002",
        title="Unrelated Paper",
        url="https://arxiv.org/abs/2401.00002",
        source_type="paper; official software",
    )
    target_update = merge_candidate_row(
        candidate_id="agent-target",
        title="Target System Paper",
        url=(
            "https://proceedings.example.test/target-paper; "
            "https://github.com/example/shared-suite"
        ),
        source_type="conference paper; official software",
    )
    unrelated_update = merge_candidate_row(
        candidate_id="agent-unrelated",
        title="Unrelated Paper",
        url=(
            "https://arxiv.org/abs/2401.00002; "
            "https://github.com/example/shared-suite"
        ),
        source_type="paper; official software",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [survivor, unrelated],
        [target_update, unrelated_update],
    )
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0003",
                surviving_candidate_id="C0001",
                reason="Implementation accompanies the target paper.",
                evidence=(
                    "https://proceedings.example.test/target-paper; "
                    "https://github.com/example/shared-suite"
                ),
            )
        ],
    )

    merged, _ = merge_candidate_files(
        existing_path, agent_paths, aliases_path=aliases_path
    )

    assert [row["candidate_id"] for row in merged] == ["C0001", "C0002"]


def test_dry_run_reports_retirement_identity_matches(tmp_path, capsys):
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Stable System Paper",
                url="https://proceedings.example.test/system-paper",
                source_type="conference paper",
            )
        ],
        [
            merge_candidate_row(
                candidate_id="agent-repository",
                title="Official System Repository",
                url="https://github.com/example/system",
                source_type="official repository",
                discovery_stream="repository",
            )
        ],
    )
    aliases_path = tmp_path / "aliases.csv"
    write_alias_rows(
        aliases_path,
        [
            alias_row(
                retired_candidate_id="C0002",
                surviving_candidate_id="C0001",
                reason="Repository implements the stable system paper.",
                evidence=(
                    "https://github.com/example/system; "
                    "https://proceedings.example.test/system-paper"
                ),
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--aliases",
            str(aliases_path),
            "--agent-file",
            str(agent_paths[0]),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "duplicate_total=1" in output
    assert "identity_matches[retirement]=1" in output
    assert "duplicate_matches[retirement][repository]=1" in output


def test_correction_applies_after_merge_and_resolves_conflict(tmp_path):
    old_url = "https://example.test/old-record"
    new_url = "https://example.test/canonical-record"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Corrected Track Paper",
        doi="10.1000/corrected-track",
        url=old_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-corrected",
        title="Corrected Track Paper",
        doi="10.1000/corrected-track",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    corrections_path = tmp_path / "corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official registry record",
                evidence="https://example.test/evidence",
                resolver="metadata-auditor",
            )
        ],
    )

    merged, conflicts = merge_candidate_files(
        existing_path,
        agent_paths,
        corrections_path=corrections_path,
    )

    assert merged[0]["url"] == new_url
    url_conflicts = [row for row in conflicts if row["field"] == "url"]
    assert len(url_conflicts) == 1
    conflict = url_conflicts[0]
    assert {conflict["value_a"], conflict["value_b"]} == {
        old_url,
        new_url,
    }
    assert conflict["resolution"] == new_url
    assert conflict["resolver"] == "metadata-auditor"
    assert (
        f"value_a={agent_paths[0].resolve().as_posix()}#agent-corrected"
        in conflict["resolution_evidence"]
        or f"value_b={agent_paths[0].resolve().as_posix()}#agent-corrected"
        in conflict["resolution_evidence"]
    )
    assert (
        f"value_a={existing_path.resolve().as_posix()}#C0001"
        in conflict["resolution_evidence"]
        or f"value_b={existing_path.resolve().as_posix()}#C0001"
        in conflict["resolution_evidence"]
    )
    assert "official registry record" in conflict["resolution_evidence"]
    assert "https://example.test/evidence" in conflict["resolution_evidence"]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {"candidate_id": "C1"},
            "candidate_id.*not a stable candidate ID",
        ),
        (
            {"candidate_id": "C0099"},
            "candidate C0099 does not exist",
        ),
        (
            {"field": "screening_status"},
            "field.*must be bibliographic",
        ),
        (
            {"old_value": "NR"},
            "old_value and new_value must be nonempty",
        ),
        (
            {"new_value": ""},
            "old_value and new_value must be nonempty",
        ),
        (
            {"old_value": "same", "new_value": "same"},
            "old_value and new_value must differ",
        ),
        (
            {
                "field": "doi",
                "old_value": "doi:10.1000/SAME",
                "new_value": "https://doi.org/10.1000/same/",
            },
            "old_value and new_value must differ",
        ),
        (
            {
                "field": "title",
                "old_value": "Café Track: Study",
                "new_value": "CAFE TRACK STUDY",
            },
            "old_value and new_value must differ",
        ),
        (
            {"reason": ""},
            "reason, evidence, and resolver must be nonempty",
        ),
        (
            {"evidence": "NR"},
            "reason, evidence, and resolver must be nonempty",
        ),
        (
            {"resolver": ""},
            "reason, evidence, and resolver must be nonempty",
        ),
    ],
)
def test_correction_validation_rejects_invalid_rows(tmp_path, updates, match):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Correction Target",
                url="https://example.test/old",
            )
        ],
    )
    row = correction_row(
        candidate_id="C0001",
        field="url",
        old_value="https://example.test/old",
        new_value="https://example.test/new",
        reason="official correction",
        evidence="https://example.test/evidence",
        resolver="reviewer",
    )
    row.update(updates)
    corrections_path = tmp_path / "corrections.csv"
    write_correction_rows(corrections_path, [row])

    with pytest.raises(ValueError, match=match):
        merge_candidate_files(
            existing_path, [], corrections_path=corrections_path
        )


def test_correction_validation_requires_exact_header(tmp_path):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [merge_candidate_row(candidate_id="C0001", title="Target")],
    )
    corrections_path = tmp_path / "corrections.csv"
    row = correction_row(
        candidate_id="C0001",
        field="title",
        old_value="Target",
        new_value="Corrected Target",
        reason="official correction",
        evidence="https://example.test/evidence",
        resolver="reviewer",
    )
    write_correction_rows(
        corrections_path,
        [row],
        header=(
            "field",
            "candidate_id",
            "old_value",
            "new_value",
            "reason",
            "evidence",
            "resolver",
        ),
    )

    with pytest.raises(ValueError, match="correction columns must be exactly"):
        merge_candidate_files(
            existing_path, [], corrections_path=corrections_path
        )


def test_correction_validation_rejects_duplicate_candidate_field(tmp_path):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Target",
                url="https://example.test/old",
            )
        ],
    )
    first = correction_row(
        candidate_id="C0001",
        field="url",
        old_value="https://example.test/old",
        new_value="https://example.test/new",
        reason="first correction",
        evidence="https://example.test/first",
        resolver="reviewer",
    )
    second = dict(first, new_value="https://example.test/other")
    corrections_path = tmp_path / "corrections.csv"
    write_correction_rows(corrections_path, [first, second])

    with pytest.raises(ValueError, match="duplicate correction.*C0001.url"):
        merge_candidate_files(
            existing_path, [], corrections_path=corrections_path
        )


def test_already_corrected_candidate_still_emits_reviewed_conflict(tmp_path):
    old_url = "https://example.test/old"
    new_url = "https://example.test/new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Already Corrected",
        doi="10.1000/already-corrected",
        url=new_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-current",
        title="Already Corrected",
        doi="10.1000/already-corrected",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    corrections_path = tmp_path / "corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )

    merged, conflicts = merge_candidate_files(
        existing_path,
        agent_paths,
        corrections_path=corrections_path,
    )

    assert merged[0]["url"] == new_url
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert {conflict["value_a"], conflict["value_b"]} == {
        old_url,
        new_url,
    }
    assert conflict["resolution"] == new_url
    assert conflict["resolver"] == "reviewer"
    assert "https://example.test/evidence" in conflict["resolution_evidence"]
    assert "@sha256:" in conflict["resolution_evidence"]


def test_correction_rejects_current_value_matching_neither_old_nor_new(
    tmp_path,
):
    existing_path, _ = build_merge_fixture(
        tmp_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Unexpected Current Value",
                url="https://example.test/third",
            )
        ],
    )
    corrections_path = tmp_path / "corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value="https://example.test/old",
                new_value="https://example.test/new",
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="C0001.url expected old or new value.*third",
    ):
        merge_candidate_files(
            existing_path, [], corrections_path=corrections_path
        )


def test_cli_applies_explicit_correction_ledger(tmp_path):
    old_url = "https://example.test/old"
    new_url = "https://example.test/new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="CLI Correction",
        doi="10.1000/cli-correction",
        url=old_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-current",
        title="CLI Correction",
        doi="10.1000/cli-correction",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    corrections_path = tmp_path / "reviewed-corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )

    assert merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent-file",
            str(agent_paths[0]),
            "--corrections",
            str(corrections_path),
            "--replace-conflicts",
            "--write",
        ]
    ) == 0

    with existing_path.open(encoding="utf-8", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    conflicts = read_conflict_rows(tmp_path / "conflicts.csv")
    assert candidate["url"] == new_url
    assert len(conflicts) == 1
    assert conflicts[0]["resolution"] == new_url
    assert conflicts[0]["resolver"] == "reviewer"


def test_replace_conflicts_correction_rerun_is_byte_idempotent(tmp_path):
    old_url = "https://example.test/old"
    new_url = "https://example.test/new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Durable Correction",
        doi="10.1000/durable-correction",
        url=old_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-current",
        title="Durable Correction",
        doi="10.1000/durable-correction",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    write_correction_rows(
        tmp_path / "candidate_corrections.csv",
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--replace-conflicts",
        "--write",
    ]

    assert merge_candidates_main(arguments) == 0
    conflicts_path = tmp_path / "conflicts.csv"
    first_candidates = existing_path.read_bytes()
    first_conflicts = conflicts_path.read_bytes()

    assert merge_candidates_main(arguments) == 0

    assert existing_path.read_bytes() == first_candidates
    assert conflicts_path.read_bytes() == first_conflicts
    conflict = read_conflict_rows(conflicts_path)[0]
    origins = {
        item
        for item in conflict["resolution_evidence"].split("; ")
        if item.startswith(("value_a=", "value_b="))
    }
    assert origins
    assert all("@sha256:" in item for item in origins)


def test_normal_rerun_preserves_reviewed_correction_conflict(tmp_path):
    old_url = "https://example.test/old"
    new_url = "https://example.test/new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Reviewed Correction",
        doi="10.1000/reviewed-correction",
        url=old_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-current",
        title="Reviewed Correction",
        doi="10.1000/reviewed-correction",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    corrections_path = tmp_path / "candidate_corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )
    common = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--corrections",
        str(corrections_path),
        "--write",
    ]
    assert merge_candidates_main([*common, "--replace-conflicts"]) == 0
    conflicts_path = tmp_path / "conflicts.csv"
    reviewed = read_conflict_rows(conflicts_path)[0]
    reviewed["conflict_id"] = "REVIEWED-CORRECTION"
    reviewed["resolution_evidence"] += "; curator review note"
    write_conflict_rows(conflicts_path, [reviewed])

    assert merge_candidates_main(common) == 0

    result = read_conflict_rows(conflicts_path)[0]
    assert result["conflict_id"] == "REVIEWED-CORRECTION"
    assert result["resolution"] == new_url
    assert result["resolver"] == "reviewer"
    assert "curator review note" in result["resolution_evidence"]
    assert "official correction" in result["resolution_evidence"]


def test_correction_merge_is_invariant_to_agent_file_order(tmp_path):
    old_url = "https://example.test/old"
    new_url = "https://example.test/new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Order Stable Correction",
        doi="10.1000/order-stable-correction",
        url=old_url,
    )
    corrected = merge_candidate_row(
        candidate_id="agent-current",
        title="Order Stable Correction",
        doi="10.1000/order-stable-correction",
        url=new_url,
    )
    unrelated = merge_candidate_row(
        candidate_id="agent-unrelated",
        title="Unrelated Candidate",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [corrected], [unrelated]
    )
    corrections_path = tmp_path / "candidate_corrections.csv"
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/evidence",
                resolver="reviewer",
            )
        ],
    )

    forward = merge_candidate_files(
        existing_path,
        agent_paths,
        corrections_path=corrections_path,
    )
    reverse = merge_candidate_files(
        existing_path,
        list(reversed(agent_paths)),
        corrections_path=corrections_path,
    )

    assert reverse == forward


def test_correction_materializes_current_conflict_for_third_value(tmp_path):
    data_dir = tmp_path / "paper" / "data"
    existing_path = data_dir / "candidates.csv"
    agent_path = data_dir / "agent_runs" / "fixture.csv"
    corrections_path = data_dir / "candidate_corrections.csv"
    old_url = "https://proceedings.mlr.press/v123/o2020a.html"
    new_url = "https://proceedings.mlr.press/v123/o-kelly20a.html"
    repository_url = "https://github.com/f1tenth/f1tenth_gym"
    existing = merge_candidate_row(
        candidate_id="C0180",
        title="F1TENTH Gym",
        url=old_url,
    )
    corrected = merge_candidate_row(
        candidate_id="agent-pmlr",
        title="F1TENTH Gym",
        url=new_url,
    )
    repository = merge_candidate_row(
        candidate_id="agent-repository",
        title="F1TENTH Gym",
        url=repository_url,
    )
    write_candidate_rows(existing_path, [existing])
    write_candidate_rows(agent_path, [corrected, repository])
    write_correction_rows(
        corrections_path,
        [
            correction_row(
                candidate_id="C0180",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence=new_url,
                resolver="entity-resolution-audit",
            )
        ],
    )
    common = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_path),
        "--corrections",
        str(corrections_path),
        "--write",
    ]

    assert merge_candidates_main([*common, "--replace-conflicts"]) == 0
    conflicts_path = data_dir / "conflicts.csv"
    first_candidates = existing_path.read_bytes()
    first_conflicts = conflicts_path.read_bytes()
    pairs = {
        frozenset((row["value_a"], row["value_b"]))
        for row in read_conflict_rows(conflicts_path)
        if row["field"] == "url"
    }
    assert pairs == {
        frozenset((old_url, new_url)),
        frozenset((old_url, repository_url)),
        frozenset((new_url, repository_url)),
    }

    assert merge_candidates_main([*common, "--replace-conflicts"]) == 0
    assert existing_path.read_bytes() == first_candidates
    assert conflicts_path.read_bytes() == first_conflicts


def test_fe3e44e_replay_reproduces_production_ledgers(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    production_data = repository / "paper" / "data"
    staging_data = tmp_path / "paper" / "data"
    staging_data.mkdir(parents=True)
    baseline = subprocess.run(
        ["git", "show", "fe3e44e:paper/data/candidates.csv"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    staged_candidates = staging_data / "candidates.csv"
    staged_aliases = staging_data / "candidate_aliases.csv"
    staged_corrections = staging_data / "candidate_corrections.csv"
    staged_candidates.write_bytes(baseline)
    staged_aliases.write_bytes(
        (production_data / "candidate_aliases.csv").read_bytes()
    )
    staged_corrections.write_bytes(
        (production_data / "candidate_corrections.csv").read_bytes()
    )
    agent_paths = [
        production_data / "agent_runs" / "aware-geometry-rl.csv",
        production_data / "agent_runs" / "aware-simulation-benchmarks.csv",
        production_data / "agent_runs" / "blind-aerial-maritime.csv",
        production_data / "agent_runs" / "blind-ground.csv",
    ]
    arguments = [
        "--existing",
        str(staged_candidates),
        "--aliases",
        str(staged_aliases),
        "--corrections",
        str(staged_corrections),
    ]
    for path in agent_paths:
        arguments.extend(("--agent-file", str(path)))
    arguments.extend(("--replace-conflicts", "--write"))

    assert merge_candidates_main(arguments) == 0
    staged_conflicts = staging_data / "conflicts.csv"
    first_candidates = staged_candidates.read_bytes()
    first_conflicts = staged_conflicts.read_bytes()
    assert any(
        row["conflict_id"] == "X11BEF7FC371E"
        for row in read_conflict_rows(staged_conflicts)
    )

    assert merge_candidates_main(arguments) == 0

    assert staged_candidates.read_bytes() == first_candidates
    assert staged_conflicts.read_bytes() == first_conflicts
    assert first_candidates == (
        production_data / "candidates.csv"
    ).read_bytes()
    assert first_conflicts == (
        production_data / "conflicts.csv"
    ).read_bytes()


def test_replace_conflicts_preserves_unresolved_hashed_origins(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Unresolved Replacement",
        authors="Original Author",
        doi="10.1000/unresolved-replacement",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-conflict",
        title="Unresolved Replacement",
        authors="Incoming Author",
        doi="10.1000/unresolved-replacement",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--replace-conflicts",
        "--write",
    ]

    assert merge_candidates_main(arguments) == 0
    conflicts_path = tmp_path / "conflicts.csv"
    first_conflicts = conflicts_path.read_bytes()
    historical_origin = immutable_origin(existing_path, existing)

    assert merge_candidates_main(arguments) == 0

    assert conflicts_path.read_bytes() == first_conflicts
    assert historical_origin in read_conflict_rows(conflicts_path)[0][
        "resolution_evidence"
    ]


def test_replace_conflicts_preserves_compatible_reviewed_resolution(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Reviewed Replacement",
        authors="Original Author",
        doi="10.1000/reviewed-replacement",
    )
    incoming = merge_candidate_row(
        candidate_id="agent-conflict",
        title="Reviewed Replacement",
        authors="Incoming Author",
        doi="10.1000/reviewed-replacement",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--replace-conflicts",
        "--write",
    ]
    assert merge_candidates_main(arguments) == 0
    conflicts_path = tmp_path / "conflicts.csv"
    reviewed = read_conflict_rows(conflicts_path)[0]
    reviewed["resolution"] = "Use the reviewed authority form."
    reviewed["resolver"] = "metadata-reviewer"
    reviewed["resolution_evidence"] += "; reviewed registry snapshot"
    write_conflict_rows(conflicts_path, [reviewed])

    assert merge_candidates_main(arguments) == 0

    result = read_conflict_rows(conflicts_path)[0]
    assert result["resolution"] == "Use the reviewed authority form."
    assert result["resolver"] == "metadata-reviewer"
    assert "reviewed registry snapshot" in result["resolution_evidence"]


@pytest.mark.parametrize(
    (
        "prior_resolution",
        "prior_resolver",
        "prior_note",
        "expected_resolver",
        "preserves_prior_note",
    ),
    [
        pytest.param(
            "",
            "resolver-only-reviewer",
            "resolver-only review note",
            "correction-reviewer",
            False,
            id="resolver-only",
        ),
        pytest.param(
            "",
            "",
            "evidence-only review note",
            "correction-reviewer",
            False,
            id="evidence-only",
        ),
        pytest.param(
            "compatible",
            "compatible-reviewer",
            "compatible review note",
            "compatible-reviewer",
            True,
            id="complete-compatible",
        ),
        pytest.param(
            "incompatible",
            "incompatible-reviewer",
            "incompatible review note",
            "correction-reviewer",
            False,
            id="complete-incompatible",
        ),
    ],
)
def test_replace_conflicts_only_preserves_complete_compatible_review(
    tmp_path,
    prior_resolution,
    prior_resolver,
    prior_note,
    expected_resolver,
    preserves_prior_note,
):
    old_url = "https://example.test/review-old"
    new_url = "https://example.test/review-new"
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Complete Review Payload",
        doi="10.1000/complete-review-payload",
        url=old_url,
    )
    incoming = merge_candidate_row(
        candidate_id="agent-current",
        title="Complete Review Payload",
        doi="10.1000/complete-review-payload",
        url=new_url,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )
    write_correction_rows(
        tmp_path / "candidate_corrections.csv",
        [
            correction_row(
                candidate_id="C0001",
                field="url",
                old_value=old_url,
                new_value=new_url,
                reason="official correction",
                evidence="https://example.test/review-evidence",
                resolver="correction-reviewer",
            )
        ],
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent-file",
        str(agent_paths[0]),
        "--replace-conflicts",
        "--write",
    ]
    assert merge_candidates_main(arguments) == 0
    conflicts_path = tmp_path / "conflicts.csv"
    prior = read_conflict_rows(conflicts_path)[0]
    prior["resolution"] = {
        "": "",
        "compatible": new_url,
        "incompatible": "https://example.test/unrelated-review",
    }[prior_resolution]
    prior["resolver"] = prior_resolver
    historical_origin = (
        "review-ledger.csv#review-1@sha256:" + "a" * 64
    )
    prior["resolution_evidence"] += (
        f"; {prior_note}; value_a={historical_origin}"
    )
    write_conflict_rows(conflicts_path, [prior])

    assert merge_candidates_main(arguments) == 0

    result = read_conflict_rows(conflicts_path)[0]
    assert result["resolution"] == new_url
    assert result["resolver"] == expected_resolver
    assert (
        prior_note in result["resolution_evidence"]
    ) is preserves_prior_note
    assert historical_origin in result["resolution_evidence"]
    assert "official correction" in result["resolution_evidence"]
