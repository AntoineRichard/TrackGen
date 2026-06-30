from __future__ import annotations

import copy
import csv
import hashlib
import math
import stat
from pathlib import Path

import pytest
import paper.scripts.coding_reliability as coding_reliability

from paper.scripts.coding_reliability import (
    CORE_FIELDS,
    cohens_kappa,
    compare_codings,
    main,
    select_reliability_sample,
)


def evidence_row(cite_key: str, domain: str, **values: str) -> dict[str, str]:
    row = {"cite_key": cite_key, "domain": domain, "title": f"Title {cite_key}"}
    row.update(values)
    return row


def ranked_keys(rows: list[dict[str, str]], count: int) -> list[str]:
    return [
        row["cite_key"]
        for row in sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(row["cite_key"].encode("utf-8")).hexdigest(),
                row["cite_key"],
            ),
        )[:count]
    ]


def test_core_fields_are_in_contract_order():
    assert CORE_FIELDS == (
        "domain",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
        "code_status",
    )


def test_sample_is_first_domain_stratified_deterministic_and_nonmutating():
    ground = [
        evidence_row(f"G{number:02d}", "ground; adjacent", note=str(number))
        for number in range(11)
    ]
    aerial = [
        evidence_row("A00", "aerial"),
        evidence_row("A01", " ; aerial ; ground "),
    ]
    maritime = [evidence_row("M00", "maritime;ground")]
    evidence = [
        ground[7],
        aerial[1],
        ground[0],
        maritime[0],
        *ground[1:7],
        aerial[0],
        *ground[8:],
    ]
    original = copy.deepcopy(evidence)
    expected = sorted(
        [
            *ranked_keys(ground, 3),
            *ranked_keys(aerial, 2),
            *ranked_keys(maritime, 1),
        ]
    )

    selected = select_reliability_sample(evidence)
    selected_again = select_reliability_sample(list(reversed(evidence)))

    assert [row["cite_key"] for row in selected] == expected
    assert [row["cite_key"] for row in selected_again] == expected
    assert [row["cite_key"] for row in selected] == sorted(expected)
    assert len([row for row in selected if row["cite_key"].startswith("G")]) == 3
    assert {row["cite_key"] for row in selected if row["cite_key"].startswith("A")} == {
        "A00",
        "A01",
    }
    assert selected[[row["cite_key"] for row in selected].index("A01")]["domain"] == (
        " ; aerial ; ground "
    )
    assert evidence == original


def test_sample_uses_cite_key_as_hash_tie_breaker(monkeypatch):
    class ConstantHash:
        def hexdigest(self) -> str:
            return "same-digest"

    monkeypatch.setattr(
        coding_reliability.hashlib,
        "sha256",
        lambda _value: ConstantHash(),
    )
    evidence = [
        evidence_row("C", "ground"),
        evidence_row("A", "ground"),
        evidence_row("B", "ground"),
    ]

    selected = select_reliability_sample(evidence, fraction=0.5)

    assert [row["cite_key"] for row in selected] == ["A", "B"]


@pytest.mark.parametrize("fraction", [0, -0.01, 1.01, math.inf, math.nan])
def test_sample_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        select_reliability_sample([evidence_row("A", "ground")], fraction=fraction)


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ([evidence_row("", "ground")], "cite_key"),
        ([evidence_row("  ", "ground")], "cite_key"),
        ([{"domain": "ground"}], "cite_key"),
        ([evidence_row("A", "")], "domain"),
        ([evidence_row("A", " ; ; ")], "domain"),
        ([{"cite_key": "A"}], "domain"),
        ([evidence_row("A", "ground"), evidence_row("A", "aerial")], "duplicate"),
    ],
)
def test_sample_rejects_invalid_identifiers_and_domains(evidence, message):
    with pytest.raises(ValueError, match=message):
        select_reliability_sample(evidence)


def coding_row(cite_key: str, **values: str) -> dict[str, str]:
    row = {"cite_key": cite_key}
    row.update(dict.fromkeys(CORE_FIELDS, "shared"))
    row.update(values)
    return row


def summary_by_field(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["field"]: row for row in rows}


def test_compare_reports_perfect_agreement_in_contract_order():
    primary = [
        coding_row("B", **dict.fromkeys(CORE_FIELDS, "beta")),
        coding_row("A", **dict.fromkeys(CORE_FIELDS, "alpha")),
    ]
    reliability = list(reversed(copy.deepcopy(primary)))

    summary = compare_codings(primary, reliability)

    assert summary == [
        {
            "field": field,
            "n": "2",
            "agreement": "1.000000",
            "kappa": "1.000000",
            "passes": "true",
        }
        for field in CORE_FIELDS
    ]


def test_compare_canonicalizes_semicolon_labels_order_insensitively():
    primary = [
        coding_row(
            "A",
            **dict.fromkeys(CORE_FIELDS, " beta ; ; alpha "),
        )
    ]
    reliability = [
        coding_row(
            "A",
            **dict.fromkeys(CORE_FIELDS, "alpha;beta"),
        )
    ]

    summary = compare_codings(primary, reliability)

    assert all(row["agreement"] == "1.000000" for row in summary)
    assert all(row["kappa"] == "NR" for row in summary)
    assert all(row["passes"] == "true" for row in summary)


def test_compare_computes_two_category_kappa():
    primary = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="alpha"),
        coding_row("C", domain="beta"),
        coding_row("D", domain="beta"),
    ]
    reliability = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="beta"),
        coding_row("C", domain="beta"),
        coding_row("D", domain="beta"),
    ]

    domain = summary_by_field(compare_codings(primary, reliability))["domain"]

    assert domain == {
        "field": "domain",
        "n": "4",
        "agreement": "0.750000",
        "kappa": "0.500000",
        "passes": "false",
    }


def test_compare_passes_at_exactly_point_eight():
    primary = [
        coding_row(
            str(number),
            domain="alpha" if number < 3 else "beta",
            generator_family="generator-a",
        )
        for number in range(5)
    ]
    reliability = copy.deepcopy(primary)
    reliability[0]["domain"] = "beta"
    reliability[0]["generator_family"] = "generator-b"
    reliability[1]["generator_family"] = "generator-b"

    summary = summary_by_field(compare_codings(primary, reliability))

    assert summary["domain"]["agreement"] == "0.800000"
    assert summary["domain"]["passes"] == "true"
    assert summary["generator_family"]["agreement"] == "0.600000"
    assert summary["generator_family"]["passes"] == "false"


def test_cohens_kappa_handles_degenerate_expected_agreement():
    assert cohens_kappa(["same", "same"], ["same", "same"]) == 1.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([], []),
        (["a"], []),
        (["a"], ["a", "b"]),
    ],
)
def test_cohens_kappa_rejects_empty_or_unequal_inputs(left, right):
    with pytest.raises(ValueError, match="equal nonzero length"):
        cohens_kappa(left, right)


def test_compare_rejects_sample_key_mismatch():
    with pytest.raises(ValueError, match="coding samples differ"):
        compare_codings([coding_row("A")], [coding_row("B")])


@pytest.mark.parametrize("side", ["primary", "reliability"])
@pytest.mark.parametrize("problem", ["blank", "missing", "duplicate"])
def test_compare_rejects_invalid_cite_keys(side, problem):
    if problem == "blank":
        invalid = [coding_row("  ")]
    elif problem == "missing":
        row = coding_row("A")
        del row["cite_key"]
        invalid = [row]
    else:
        invalid = [coding_row("A"), coding_row("A")]
    valid = [coding_row("A")]
    primary = invalid if side == "primary" else valid
    reliability = invalid if side == "reliability" else valid

    with pytest.raises(ValueError, match="cite_key|duplicate"):
        compare_codings(primary, reliability)


@pytest.mark.parametrize("side", ["primary", "reliability"])
def test_compare_rejects_missing_core_fields(side):
    invalid = coding_row("A")
    del invalid["validity_strategy"]
    valid = coding_row("A")
    primary = [invalid] if side == "primary" else [valid]
    reliability = [invalid] if side == "reliability" else [valid]

    with pytest.raises(ValueError, match="validity_strategy"):
        compare_codings(primary, reliability)


@pytest.mark.parametrize(
    ("primary", "reliability"),
    [
        ([], []),
        ([], [coding_row("A")]),
        ([coding_row("A")], []),
    ],
)
def test_compare_rejects_empty_inputs(primary, reliability):
    with pytest.raises(ValueError, match="nonempty"):
        compare_codings(primary, reliability)


def write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def staged_files(output: Path) -> list[Path]:
    return list(output.parent.glob(f".{output.name}.*.tmp"))


def test_prepare_cli_preserves_header_complete_rows_and_utf8(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "coding_primary.csv"
    header = ("title", "cite_key", "domain", "notes")
    evidence = [
        {
            "title": "Title C",
            "cite_key": "C",
            "domain": "ground",
            "notes": "\u00e9lan",
        },
        {
            "title": "Title A",
            "cite_key": "A",
            "domain": "ground;adjacent",
            "notes": "alpha",
        },
        {
            "title": "Title B",
            "cite_key": "B",
            "domain": "ground",
            "notes": "beta",
        },
    ]
    write_csv(evidence_path, header, evidence)
    original_input = evidence_path.read_bytes()
    expected_keys = sorted(ranked_keys(evidence, 2))
    expected_by_key = {row["cite_key"]: row for row in evidence}

    result = main(
        [
            "--prepare",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    actual_header, actual_rows = read_csv(output_path)
    assert result == 0
    assert actual_header == header
    assert actual_rows == [expected_by_key[key] for key in expected_keys]
    assert output_path.read_bytes().decode("utf-8")
    assert evidence_path.read_bytes() == original_input
    assert not staged_files(output_path)


def test_compare_cli_writes_summary_csv(tmp_path):
    primary_path = tmp_path / "coding_primary.csv"
    reliability_path = tmp_path / "coding_reliability.csv"
    output_path = tmp_path / "coding_reliability_summary.csv"
    header = ("cite_key", *CORE_FIELDS)
    primary = [
        coding_row("B", domain="ground; aerial"),
        coding_row("A", domain="maritime"),
    ]
    reliability = [
        coding_row("A", domain="maritime"),
        coding_row("B", domain="aerial;ground"),
    ]
    write_csv(primary_path, header, primary)
    write_csv(reliability_path, header, reliability)

    result = main(
        [
            "--primary",
            str(primary_path),
            "--reliability",
            str(reliability_path),
            "--output",
            str(output_path),
        ]
    )

    actual_header, actual_rows = read_csv(output_path)
    assert result == 0
    assert actual_header == ("field", "n", "agreement", "kappa", "passes")
    assert actual_rows == [
        {
            "field": field,
            "n": "2",
            "agreement": "1.000000",
            "kappa": "1.000000" if field == "domain" else "NR",
            "passes": "true",
        }
        for field in CORE_FIELDS
    ]
    assert not staged_files(output_path)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--output", "OUTPUT"],
        ["--prepare", "--output", "OUTPUT"],
        ["--prepare", "--evidence", "INPUT"],
        ["--evidence", "INPUT", "--output", "OUTPUT"],
        ["--primary", "INPUT", "--output", "OUTPUT"],
        ["--reliability", "INPUT", "--output", "OUTPUT"],
        [
            "--prepare",
            "--evidence",
            "INPUT",
            "--primary",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        [
            "--primary",
            "INPUT",
            "--reliability",
            "INPUT",
            "--evidence",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        ["--output", "OUTPUT", "--bogus"],
        ["--output", "OUTPUT", "extra"],
    ],
)
def test_cli_rejects_missing_mixed_and_extraneous_mode_arguments(
    tmp_path,
    arguments,
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    resolved = [
        str(input_path)
        if argument == "INPUT"
        else str(output_path)
        if argument == "OUTPUT"
        else argument
        for argument in arguments
    ]

    with pytest.raises(SystemExit) as exc_info:
        main(resolved)

    assert exc_info.value.code == 2
    assert not output_path.exists()
    assert not staged_files(output_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"", "header"),
        (b"A,ground\nB,aerial\n", "required columns"),
    ],
)
def test_prepare_cli_rejects_empty_and_headerless_csv(
    tmp_path,
    contents,
    message,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    evidence_path.write_bytes(contents)
    output_path.write_bytes(b"existing output\n")
    original_output = output_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        main(
            [
                "--prepare",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


def test_prepare_cli_rejects_header_only_csv(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(evidence_path, ("cite_key", "domain", "title"), [])
    output_path.write_bytes(b"existing output\n")
    original_output = output_path.read_bytes()

    with pytest.raises(ValueError, match="at least one data row"):
        main(
            [
                "--prepare",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


def test_prepare_validation_failure_preserves_output_and_cleans_staging(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [
            {"cite_key": "A", "domain": "ground"},
            {"cite_key": "A", "domain": "aerial"},
        ],
    )
    output_path.write_bytes(b"existing output\r\n")
    original_output = output_path.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        main(
            [
                "--prepare",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


def test_compare_failure_preserves_output_and_cleans_staging(tmp_path):
    primary_path = tmp_path / "primary.csv"
    reliability_path = tmp_path / "reliability.csv"
    output_path = tmp_path / "output.csv"
    header = ("cite_key", *CORE_FIELDS)
    write_csv(primary_path, header, [coding_row("A")])
    write_csv(reliability_path, header, [coding_row("B")])
    output_path.write_bytes(b"existing output\r\n")
    original_output = output_path.read_bytes()

    with pytest.raises(ValueError, match="coding samples differ"):
        main(
            [
                "--primary",
                str(primary_path),
                "--reliability",
                str(reliability_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


@pytest.mark.parametrize("cite_key", [" A", "A ", "\tA", "A\n"])
def test_sample_rejects_cite_key_surrounding_whitespace(cite_key):
    with pytest.raises(ValueError, match="cite_key.*surrounding whitespace"):
        select_reliability_sample([evidence_row(cite_key, "ground")])


def test_sample_rejects_padded_raw_key_collision():
    evidence = [
        evidence_row("A", "ground"),
        evidence_row(" A", "ground"),
    ]

    with pytest.raises(ValueError, match="cite_key.*surrounding whitespace"):
        select_reliability_sample(evidence)


def test_sample_retains_distinct_exact_unpadded_keys():
    evidence = [
        evidence_row("A", "ground"),
        evidence_row("a", "ground"),
        evidence_row("A B", "ground"),
    ]

    selected = select_reliability_sample(evidence, fraction=1.0)

    assert [row["cite_key"] for row in selected] == ["A", "A B", "a"]


@pytest.mark.parametrize("side", ["primary", "reliability"])
def test_compare_rejects_padded_raw_key_collision(side):
    invalid = [coding_row("A"), coding_row(" A")]
    valid = [coding_row("A"), coding_row("B")]
    primary = invalid if side == "primary" else valid
    reliability = invalid if side == "reliability" else valid

    with pytest.raises(
        ValueError,
        match=rf"{side}.*cite_key.*surrounding whitespace",
    ):
        compare_codings(primary, reliability)


def test_compare_retains_distinct_exact_unpadded_keys():
    rows = [coding_row("A"), coding_row("a"), coding_row("A B")]

    summary = compare_codings(rows, list(reversed(copy.deepcopy(rows))))

    assert all(row["n"] == "3" for row in summary)


@pytest.mark.parametrize("side", ["primary", "reliability"])
@pytest.mark.parametrize("field", CORE_FIELDS)
@pytest.mark.parametrize("value", ["", "   ", ";", " ; ; "])
def test_compare_rejects_canonical_empty_core_values(side, field, value):
    invalid = coding_row("A", **{field: value})
    valid = coding_row("A")
    primary = [invalid] if side == "primary" else [valid]
    reliability = [invalid] if side == "reliability" else [valid]

    with pytest.raises(ValueError, match=rf"{side}.*{field}.*nonempty"):
        compare_codings(primary, reliability)


@pytest.mark.parametrize("side", ["primary", "reliability"])
def test_empty_core_value_compare_failure_preserves_existing_output(tmp_path, side):
    primary_path = tmp_path / "primary.csv"
    reliability_path = tmp_path / "reliability.csv"
    output_path = tmp_path / "output.csv"
    header = ("cite_key", *CORE_FIELDS)
    primary = [coding_row("A")]
    reliability = [coding_row("A")]
    target = primary if side == "primary" else reliability
    target[0]["generator_family"] = " ; ; "
    write_csv(primary_path, header, primary)
    write_csv(reliability_path, header, reliability)
    output_path.write_bytes(b"existing output\r\n")
    original_output = output_path.read_bytes()

    with pytest.raises(
        ValueError,
        match=rf"{side}.*generator_family.*nonempty",
    ):
        main(
            [
                "--primary",
                str(primary_path),
                "--reliability",
                str(reliability_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


def test_prepare_cli_rejects_whitespace_header_and_preserves_output(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain", "   "),
        [{"cite_key": "A", "domain": "ground", "   ": "value"}],
    )
    output_path.write_bytes(b"existing output\r\n")
    original_output = output_path.read_bytes()

    with pytest.raises(ValueError, match="blank column name"):
        main(
            [
                "--prepare",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not staged_files(output_path)


def test_compare_reports_nr_when_both_coders_use_one_shared_category():
    rows = [coding_row("A"), coding_row("B")]

    summary = compare_codings(rows, copy.deepcopy(rows))

    assert all(row["agreement"] == "1.000000" for row in summary)
    assert all(row["kappa"] == "NR" for row in summary)
    assert all(row["passes"] == "true" for row in summary)


@pytest.mark.parametrize("many_side", ["primary", "reliability"])
def test_compare_reports_nr_for_asymmetric_one_vs_many_categories(many_side):
    one_category = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="alpha"),
    ]
    many_categories = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="beta"),
    ]
    primary = many_categories if many_side == "primary" else one_category
    reliability = many_categories if many_side == "reliability" else one_category

    domain = summary_by_field(compare_codings(primary, reliability))["domain"]

    assert domain == {
        "field": "domain",
        "n": "2",
        "agreement": "0.500000",
        "kappa": "NR",
        "passes": "false",
    }


def test_compare_computes_kappa_when_multi_category_coders_share_one_category():
    primary = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="beta"),
    ]
    reliability = [
        coding_row("A", domain="alpha"),
        coding_row("B", domain="gamma"),
    ]

    domain = summary_by_field(compare_codings(primary, reliability))["domain"]

    assert domain["agreement"] == "0.500000"
    assert domain["kappa"] == "0.333333"
    assert domain["passes"] == "false"


def test_new_output_receives_deterministic_repository_mode(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )

    main(
        [
            "--prepare",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o644


def test_existing_output_mode_is_preserved(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"existing output\n")
    output_path.chmod(0o640)

    main(
        [
            "--prepare",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o640


def test_prepare_output_has_exact_lf_bytes(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain", "title"),
        [
            {"cite_key": "B", "domain": "ground", "title": "Beta"},
            {"cite_key": "A", "domain": "ground", "title": "Alpha"},
        ],
    )

    main(
        [
            "--prepare",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.read_bytes() == (
        b"cite_key,domain,title\n"
        b"A,ground,Alpha\n"
        b"B,ground,Beta\n"
    )


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_injected_atomic_failure_preserves_output_and_cleans_staging(
    tmp_path,
    monkeypatch,
    failure_point,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "output.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"existing output\r\n")
    output_path.chmod(0o640)
    original_bytes = output_path.read_bytes()
    original_mode = stat.S_IMODE(output_path.stat().st_mode)

    def fail_writerows(_writer, _rows):
        raise OSError("injected write failure")

    real_replace = Path.replace

    def fail_replace(source, target):
        if Path(target) == output_path:
            raise OSError("injected replace failure")
        return real_replace(source, target)

    if failure_point == "write":
        monkeypatch.setattr(
            coding_reliability.csv.DictWriter,
            "writerows",
            fail_writerows,
        )
    else:
        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        main(
            [
                "--prepare",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_bytes
    assert stat.S_IMODE(output_path.stat().st_mode) == original_mode
    assert not staged_files(output_path)
