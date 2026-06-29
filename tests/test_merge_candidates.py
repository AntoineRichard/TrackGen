import csv
from pathlib import Path

import pytest

import paper.scripts.merge_candidates as merge_candidates_module

from paper.scripts.merge_candidates import (
    main as merge_candidates_main,
    merge_candidate_files,
)
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
    assert conflicts[0]["value_a"] == current
    assert conflicts[0]["value_b"] == proposed


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


def test_rerunning_merge_with_same_agent_inputs_is_idempotent(tmp_path):
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
    assert second_conflicts == first_conflicts


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
    assert conflicts[0]["resolution_evidence"] == (
        f"value_a={existing_path.resolve().as_posix()}#C0001@row:2; "
        f"value_b={agent_paths[0].resolve().as_posix()}#agent-a@row:2; "
        f"value_b={agent_paths[1].resolve().as_posix()}#agent-b@row:2"
    )


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

    merged, _ = merge_candidate_files(existing_path, agent_paths)

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

    assert read_conflict_rows(conflicts_path) == [reviewed]


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

    _, conflicts = merge_candidate_files(existing_path, agent_paths)

    author_conflict = next(row for row in conflicts if row["field"] == "authors")
    assert author_conflict["resolution_evidence"] == (
        f"value_a={agent_paths[0].resolve().as_posix()}#agent-a@row:2; "
        f"value_b={agent_paths[1].resolve().as_posix()}#agent-b@row:2"
    )
    assert "candidates.csv#C0001" not in author_conflict["resolution_evidence"]


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
    assert [
        (row["field"], row["value_a"], row["value_b"])
        for row in conflicts
    ] == [
        (
            "exclusion_reason",
            "Reviewed exclusion rationale.",
            "Different agent rationale.",
        )
    ]


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
    write_candidate_rows(
        existing_path,
        [
            merge_candidate_row(
                candidate_id="C0001",
                title="Stable Track",
                authors="Original Author",
                doi="10.1000/portable-path",
            )
        ],
    )
    write_candidate_rows(
        agent_path,
        [
            merge_candidate_row(
                candidate_id="agent-row",
                title="Stable Track",
                authors="Incoming Author",
                doi="10.1000/portable-path",
            )
        ],
    )

    _, conflicts = merge_candidate_files(existing_path, [agent_path])

    author_conflict = next(row for row in conflicts if row["field"] == "authors")
    assert author_conflict["resolution_evidence"] == (
        "value_a=paper/data/candidates.csv#C0001@row:2; "
        "value_b=paper/data/agent_runs/agent.csv#agent-row@row:2"
    )
