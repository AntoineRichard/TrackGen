from __future__ import annotations

import ast
import copy
import csv
import errno
import hashlib
import math
import os
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


def _exception_text(error: BaseException) -> str:
    return "\n".join((str(error), *getattr(error, "__notes__", ())))


def test_cleanup_source_has_no_destructive_or_overwriting_namespace_calls():
    tree = ast.parse(Path(coding_reliability.__file__).read_text(encoding="utf-8"))
    forbidden = []
    forbidden_qualified = {
        ("os", "remove"),
        ("os", "rename"),
        ("os", "replace"),
        ("os", "rmdir"),
        ("os", "unlink"),
        ("shutil", "rmtree"),
    }
    forbidden_native = {"remove", "rename", "rmdir", "unlink", "unlinkat"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (function.value.id, function.attr) in forbidden_qualified
        ):
            forbidden.append((node.lineno, f"{function.value.id}.{function.attr}"))
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"rename", "rmdir", "rmtree", "unlink"}
        ):
            forbidden.append((node.lineno, function.attr))
        if (
            isinstance(function, ast.Name)
            and function.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in forbidden_native
        ):
            forbidden.append((node.lineno, str(node.args[1].value)))
    assert forbidden == []


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
        "survey_evidence_tier",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
        "code_status",
        "asset_status",
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


def test_compare_counts_core_and_supporting_as_an_independent_disagreement():
    primary = [coding_row("A", survey_evidence_tier="core")]
    reliability = [coding_row("A", survey_evidence_tier="supporting")]

    tier = summary_by_field(compare_codings(primary, reliability))["survey_evidence_tier"]

    assert tier == {
        "field": "survey_evidence_tier",
        "n": "1",
        "agreement": "0.000000",
        "kappa": "NR",
        "passes": "false",
    }


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
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="core"),
        coding_row("C", survey_evidence_tier="supporting"),
        coding_row("D", survey_evidence_tier="supporting"),
    ]
    reliability = [
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="supporting"),
        coding_row("C", survey_evidence_tier="supporting"),
        coding_row("D", survey_evidence_tier="supporting"),
    ]

    tier = summary_by_field(compare_codings(primary, reliability))["survey_evidence_tier"]

    assert tier == {
        "field": "survey_evidence_tier",
        "n": "4",
        "agreement": "0.750000",
        "kappa": "0.500000",
        "passes": "false",
    }


def test_compare_passes_at_exactly_point_eight():
    primary = [
        coding_row(
            str(number),
            survey_evidence_tier="core" if number < 3 else "supporting",
            generator_family="generator-a",
        )
        for number in range(5)
    ]
    reliability = copy.deepcopy(primary)
    reliability[0]["survey_evidence_tier"] = "supporting"
    reliability[0]["generator_family"] = "generator-b"
    reliability[1]["generator_family"] = "generator-b"

    summary = summary_by_field(compare_codings(primary, reliability))

    assert summary["survey_evidence_tier"]["agreement"] == "0.800000"
    assert summary["survey_evidence_tier"]["passes"] == "true"
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
        coding_row("B", course_object="ground; aerial"),
        coding_row("A", course_object="maritime"),
    ]
    reliability = [
        coding_row("A", course_object="maritime"),
        coding_row("B", course_object="aerial;ground"),
    ]
    write_csv(primary_path, header, primary)
    write_csv(reliability_path, header, reliability)
    primary_header, _ = read_csv(primary_path)
    reliability_header, _ = read_csv(reliability_path)
    assert primary_header[:2] == ("cite_key", "survey_evidence_tier")
    assert reliability_header[:2] == ("cite_key", "survey_evidence_tier")


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
    assert actual_rows[0]["field"] == "survey_evidence_tier"
    assert actual_rows == [
        {
            "field": field,
            "n": "2",
            "agreement": "1.000000",
            "kappa": "1.000000" if field == "course_object" else "NR",
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
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="core"),
    ]
    many_categories = [
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="supporting"),
    ]
    primary = many_categories if many_side == "primary" else one_category
    reliability = many_categories if many_side == "reliability" else one_category

    tier = summary_by_field(compare_codings(primary, reliability))["survey_evidence_tier"]

    assert tier == {
        "field": "survey_evidence_tier",
        "n": "2",
        "agreement": "0.500000",
        "kappa": "NR",
        "passes": "false",
    }


def test_compare_computes_kappa_when_multi_category_coders_share_one_category():
    primary = [
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="supporting"),
    ]
    reliability = [
        coding_row("A", survey_evidence_tier="core"),
        coding_row("B", survey_evidence_tier="contextual"),
    ]

    tier = summary_by_field(compare_codings(primary, reliability))["survey_evidence_tier"]

    assert tier["agreement"] == "0.500000"
    assert tier["kappa"] == "0.333333"
    assert tier["passes"] == "false"


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


@pytest.mark.parametrize("failure_point", ["write", "exchange"])
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

    real_exchange = coding_reliability._rename_exchange_at
    injected = False

    def fail_exchange(directory, source_name, target_name):
        nonlocal injected
        if not injected and directory.path / target_name == output_path:
            injected = True
            raise OSError("injected exchange failure")
        return real_exchange(
            directory,
            source_name,
            target_name,
        )

    if failure_point == "write":
        monkeypatch.setattr(
            coding_reliability.csv.DictWriter,
            "writerows",
            fail_writerows,
        )
    else:
        monkeypatch.setattr(
            coding_reliability,
            "_rename_exchange_at",
            fail_exchange,
        )

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

SELECTION_FIELDS = (
    "cite_key",
    "first_domain",
    "rank_sha256",
    "evidence_sha256",
)
DEFAULT_EVIDENCE_SHA256 = hashlib.sha256(b"test evidence snapshot").hexdigest()


def selection_row(
    cite_key: str,
    first_domain: str,
    evidence_sha256: str = DEFAULT_EVIDENCE_SHA256,
) -> dict[str, str]:
    return {
        "cite_key": cite_key,
        "first_domain": first_domain,
        "rank_sha256": hashlib.sha256(cite_key.encode("utf-8")).hexdigest(),
        "evidence_sha256": evidence_sha256,
    }


def test_select_cli_writes_only_deterministic_selection_metadata(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    selection_path = tmp_path / "coding_reliability_selection.csv"
    header = (
        "cite_key",
        "domain",
        "course_object",
        "coding_notes",
    )
    ground = [
        {
            "cite_key": f"G{number:02d}",
            "domain": " ground ; adjacent ",
            "course_object": f"PRIMARY-OBJECT-{number}",
            "coding_notes": f"TECHNICAL-EVIDENCE-{number}",
        }
        for number in range(11)
    ]
    other = [
        {
            "cite_key": "AerienÉtude",
            "domain": "aerial;ground",
            "course_object": "PRIMARY-OBJECT-UNICODE",
            "coding_notes": "TECHNICAL-EVIDENCE-UNICODE",
        }
    ]
    evidence = [*reversed(ground), *other]
    write_csv(evidence_path, header, evidence)
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    expected_keys = sorted([*ranked_keys(ground, 3), "AerienÉtude"])

    result = main(
        [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(selection_path),
        ]
    )

    actual_header, actual_rows = read_csv(selection_path)
    assert result == 0
    assert actual_header == SELECTION_FIELDS
    assert actual_rows == [
        selection_row(
            cite_key,
            "aerial" if cite_key == "AerienÉtude" else "ground",
            evidence_sha256,
        )
        for cite_key in expected_keys
    ]
    output_bytes = selection_path.read_bytes()
    assert output_bytes.startswith(
        b"cite_key,first_domain,rank_sha256,evidence_sha256\n"
    )
    assert b"course_object" not in output_bytes
    assert b"PRIMARY-OBJECT" not in output_bytes
    assert b"TECHNICAL-EVIDENCE" not in output_bytes
    assert "AerienÉtude".encode() in output_bytes
    assert not staged_files(selection_path)


def test_select_cli_is_independent_of_evidence_row_order(tmp_path):
    first_evidence_path = tmp_path / "evidence-first.csv"
    second_evidence_path = tmp_path / "evidence-second.csv"
    first_selection_path = tmp_path / "selection-first.csv"
    second_selection_path = tmp_path / "selection-second.csv"
    header = ("cite_key", "domain")
    evidence = [
        {"cite_key": f"K{number:02d}", "domain": "ground"}
        for number in range(17)
    ]
    write_csv(first_evidence_path, header, evidence)
    write_csv(second_evidence_path, header, list(reversed(evidence)))

    main(
        [
            "--select",
            "--evidence",
            str(first_evidence_path),
            "--output",
            str(first_selection_path),
        ]
    )
    main(
        [
            "--select",
            "--evidence",
            str(second_evidence_path),
            "--output",
            str(second_selection_path),
        ]
    )

    first_header, first_rows = read_csv(first_selection_path)
    second_header, second_rows = read_csv(second_selection_path)
    assert first_header == SELECTION_FIELDS
    assert second_header == SELECTION_FIELDS
    assert [
        {field: value for field, value in row.items() if field != "evidence_sha256"}
        for row in first_rows
    ] == [
        {field: value for field, value in row.items() if field != "evidence_sha256"}
        for row in second_rows
    ]
    first_digest = hashlib.sha256(first_evidence_path.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second_evidence_path.read_bytes()).hexdigest()
    assert first_digest != second_digest
    assert {row["evidence_sha256"] for row in first_rows} == {first_digest}
    assert {row["evidence_sha256"] for row in second_rows} == {second_digest}


PACKET_FIELDS = (
    "candidate_id",
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
TEMPLATE_FIELDS = ("cite_key", *CORE_FIELDS)


def candidate_row(
    cite_key: str,
    candidate_id: str,
    **values: str,
) -> dict[str, str]:
    row = {
        "candidate_id": candidate_id,
        "cite_key": cite_key,
        "title": f"Title {cite_key}",
        "authors": "Doe; Roe",
        "year": "2026",
        "venue": "Test Venue",
        "doi": f"10.1000/{candidate_id}",
        "url": f"https://example.test/{candidate_id}",
        "source_type": "paper; official repository",
    }
    row.update(values)
    return row


def test_prepare_blind_writes_metadata_packet_and_blank_template(tmp_path):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "blind_packet.csv"
    template_path = tmp_path / "coding_reliability.csv"
    selection = [
        selection_row("Zeta", "PRIMARY-DOMAIN-GROUND"),
        selection_row("Éclair", "PRIMARY-DOMAIN-AERIAL"),
    ]
    write_csv(selection_path, SELECTION_FIELDS, selection)
    candidate_header = (*PACKET_FIELDS, "domain", "technical_evidence")
    selected_candidates = [
        candidate_row(
            "Éclair",
            "C0002",
            title="Trajectoire, étude",
            authors="García; Müller",
            domain="PRIMARY-DOMAIN-AERIAL",
            technical_evidence="SECRET-TECHNICAL-EVIDENCE",
        ),
        candidate_row(
            "Zeta",
            "C0001",
            title="Line one\nLine two",
            domain="PRIMARY-DOMAIN-GROUND",
            technical_evidence="SECRET-TECHNICAL-EVIDENCE",
        ),
    ]
    candidates = [
        selected_candidates[1],
        candidate_row(
            "Unselected",
            "C0003",
            domain="UNSELECTED-DOMAIN",
            technical_evidence="UNSELECTED-TECHNICAL-EVIDENCE",
        ),
        candidate_row("", "C0004"),
        selected_candidates[0],
    ]
    write_csv(candidates_path, candidate_header, candidates)

    result = main(
        [
            "--prepare-blind",
            "--selection",
            str(selection_path),
            "--candidates",
            str(candidates_path),
            "--packet-output",
            str(packet_path),
            "--template-output",
            str(template_path),
        ]
    )

    packet_header, packet_rows = read_csv(packet_path)
    template_header, template_rows = read_csv(template_path)
    expected_candidates = sorted(
        selected_candidates,
        key=lambda row: row["cite_key"],
    )
    assert result == 0
    assert packet_header == PACKET_FIELDS
    assert packet_rows == [
        {field: row[field] for field in PACKET_FIELDS}
        for row in expected_candidates
    ]
    assert template_header == TEMPLATE_FIELDS
    assert template_rows == [
        {"cite_key": row["cite_key"], **dict.fromkeys(CORE_FIELDS, "")}
        for row in expected_candidates
    ]
    assert template_header[:2] == ("cite_key", "survey_evidence_tier")
    packet_bytes = packet_path.read_bytes()
    assert "Trajectoire, étude".encode() in packet_bytes
    assert "García; Müller".encode() in packet_bytes
    assert b"first_domain" not in packet_bytes
    assert b"rank_sha256" not in packet_bytes
    assert b"evidence_sha256" not in packet_bytes
    assert DEFAULT_EVIDENCE_SHA256.encode() not in packet_bytes
    assert b"PRIMARY-DOMAIN" not in packet_bytes
    assert b"SECRET-TECHNICAL-EVIDENCE" not in packet_bytes
    assert b"UNSELECTED" not in packet_bytes
    assert not staged_files(packet_path)
    assert not staged_files(template_path)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_candidate",
        "duplicate_candidate",
        "duplicate_unselected_candidate",
        "duplicate_selection",
        "stale_selection",
        "tampered_rank",
        "selection_extra_column",
    ],
)
def test_prepare_blind_rejects_selection_candidate_mismatches_without_publication(
    tmp_path,
    problem,
):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "blind_packet.csv"
    template_path = tmp_path / "coding_reliability.csv"
    selection_header = SELECTION_FIELDS
    selection = [selection_row("A", "ground"), selection_row("B", "ground")]
    candidates = [candidate_row("A", "C0001"), candidate_row("B", "C0002")]

    if problem == "missing_candidate":
        candidates.pop()
    elif problem == "duplicate_candidate":
        candidates.append(candidate_row("A", "C0003"))
    elif problem == "duplicate_unselected_candidate":
        candidates.extend(
            [
                candidate_row("Unselected", "C0003"),
                candidate_row("Unselected", "C0004"),
            ]
        )
    elif problem == "duplicate_selection":
        selection.append(selection_row("A", "ground"))
    elif problem == "stale_selection":
        selection[1] = selection_row("RetiredKey", "ground")
    elif problem == "tampered_rank":
        selection[0]["rank_sha256"] = "0" * 64
    else:
        selection_header = (*SELECTION_FIELDS, "domain")
        for row in selection:
            row["domain"] = "PRIMARY-CODING"

    write_csv(selection_path, selection_header, selection)
    write_csv(candidates_path, PACKET_FIELDS, candidates)
    packet_path.write_bytes(b"existing packet\r\n")
    template_path.write_bytes(b"existing template\r\n")
    original_packet = packet_path.read_bytes()
    original_template = template_path.read_bytes()

    with pytest.raises(ValueError):
        main(
            [
                "--prepare-blind",
                "--selection",
                str(selection_path),
                "--candidates",
                str(candidates_path),
                "--packet-output",
                str(packet_path),
                "--template-output",
                str(template_path),
            ]
        )

    assert packet_path.read_bytes() == original_packet
    assert template_path.read_bytes() == original_template
    assert not staged_files(packet_path)
    assert not staged_files(template_path)


def complete_evidence_row(
    cite_key: str,
    domain: str,
    **values: str,
) -> dict[str, str]:
    row = coding_row(cite_key, domain=domain)
    row.update(
        {
            "title": f"Title, {cite_key}",
            "coding_notes": f"Evidence for {cite_key}",
        }
    )
    row.update(values)
    return row


def test_materialize_primary_is_delayed_and_copies_exact_evidence_rows(tmp_path):
    evidence_path = tmp_path / "evidence.csv"
    selection_path = tmp_path / "selection.csv"
    primary_path = tmp_path / "coding_primary.csv"
    header = ("cite_key", "survey_evidence_tier", "domain", *CORE_FIELDS[1:], "title", "coding_notes")
    evidence = [
        complete_evidence_row(
            "Éclair",
            "aerial;ground",
            course_object="closed; open",
            coding_notes="naïve evidence, with comma",
        ),
        complete_evidence_row("Delta", "ground"),
        complete_evidence_row("Alpha", "ground"),
        complete_evidence_row("Charlie", "ground"),
        complete_evidence_row("Bravo", "ground"),
    ]
    write_csv(evidence_path, header, evidence)
    original_evidence = evidence_path.read_bytes()
    expected = select_reliability_sample(evidence)

    main(
        [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(selection_path),
        ]
    )

    assert selection_path.is_file()
    assert not primary_path.exists()

    result = main(
        [
            "--materialize-primary",
            "--selection",
            str(selection_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(primary_path),
        ]
    )

    actual_header, actual_rows = read_csv(primary_path)
    assert result == 0
    assert actual_header == header
    assert actual_rows == expected
    assert actual_header[:2] == ("cite_key", "survey_evidence_tier")
    assert [row["cite_key"] for row in actual_rows] == sorted(
        row["cite_key"] for row in actual_rows
    )
    assert "naïve evidence, with comma".encode() in primary_path.read_bytes()
    assert evidence_path.read_bytes() == original_evidence
    assert not staged_files(primary_path)


@pytest.mark.parametrize(
    "tamper",
    [
        "rank",
        "first_domain",
        "evidence_sha256",
        "membership",
        "missing",
        "extra",
    ],
)
def test_materialize_primary_rejects_tampered_selection_and_preserves_output(
    tmp_path,
    tamper,
):
    evidence_path = tmp_path / "evidence.csv"
    selection_path = tmp_path / "selection.csv"
    primary_path = tmp_path / "coding_primary.csv"
    header = ("cite_key", "survey_evidence_tier", "domain", *CORE_FIELDS[1:], "title", "coding_notes")
    evidence = [
        complete_evidence_row(cite_key, "ground")
        for cite_key in ("A", "B", "C", "D")
    ]
    write_csv(evidence_path, header, evidence)
    main(
        [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(selection_path),
        ]
    )
    selection_header, selection = read_csv(selection_path)
    selected_keys = {row["cite_key"] for row in selection}
    unselected_key = next(
        row["cite_key"] for row in evidence if row["cite_key"] not in selected_keys
    )

    if tamper == "rank":
        selection[0]["rank_sha256"] = "f" * 64
    elif tamper == "first_domain":
        selection[0]["first_domain"] = "aerial"
    elif tamper == "evidence_sha256":
        selection[0]["evidence_sha256"] = "f" * 64
    elif tamper == "membership":
        selection[0] = selection_row(unselected_key, "ground")
    elif tamper == "missing":
        selection.pop()
    else:
        selection.append(selection_row(unselected_key, "ground"))

    write_csv(selection_path, selection_header, selection)
    primary_path.write_bytes(b"sealed existing primary\r\n")
    primary_path.chmod(0o640)
    original_bytes = primary_path.read_bytes()
    original_mode = stat.S_IMODE(primary_path.stat().st_mode)

    with pytest.raises(
        ValueError,
        match="selection|rank_sha256|first_domain|evidence_sha256",
    ):
        main(
            [
                "--materialize-primary",
                "--selection",
                str(selection_path),
                "--evidence",
                str(evidence_path),
                "--output",
                str(primary_path),
            ]
        )

    assert primary_path.read_bytes() == original_bytes
    assert stat.S_IMODE(primary_path.stat().st_mode) == original_mode
    assert not staged_files(primary_path)


@pytest.mark.parametrize(
    "drift",
    ["domain", "cite_key", "non_domain", "line_endings"],
)
def test_materialize_primary_rejects_stale_selection_after_evidence_drift(
    tmp_path,
    drift,
):
    evidence_path = tmp_path / "evidence.csv"
    selection_path = tmp_path / "selection.csv"
    primary_path = tmp_path / "coding_primary.csv"
    header = ("cite_key", "survey_evidence_tier", "domain", *CORE_FIELDS[1:], "title", "coding_notes")
    evidence = [
        complete_evidence_row("A", "ground"),
        complete_evidence_row("B", "ground"),
    ]
    write_csv(evidence_path, header, evidence)
    main(
        [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(selection_path),
        ]
    )

    if drift == "domain":
        evidence[0]["domain"] = "aerial"
    elif drift == "cite_key":
        evidence[0]["cite_key"] = "RetitledKey"
    elif drift == "non_domain":
        evidence[0]["coding_notes"] = "changed technical evidence"
    else:
        evidence_bytes = evidence_path.read_bytes()
        assert b"\r\n" not in evidence_bytes
        evidence_path.write_bytes(evidence_bytes.replace(b"\n", b"\r\n"))
    if drift != "line_endings":
        write_csv(evidence_path, header, evidence)
    primary_path.write_bytes(b"sealed existing primary\n")
    original_output = primary_path.read_bytes()

    with pytest.raises(ValueError, match="selection"):
        main(
            [
                "--materialize-primary",
                "--selection",
                str(selection_path),
                "--evidence",
                str(evidence_path),
                "--output",
                str(primary_path),
            ]
        )

    assert primary_path.read_bytes() == original_output
    assert not staged_files(primary_path)


def hidden_output_artifacts(*outputs: Path) -> list[Path]:
    prefixes = tuple(f".{output.name}." for output in outputs)
    return [
        path
        for output in outputs
        for path in output.parent.iterdir()
        if path.name.startswith(prefixes)
    ]


@pytest.mark.parametrize("failure_point", ["second_stage", "second_publish"])
def test_prepare_blind_two_output_failure_rolls_back_both_destinations(
    tmp_path,
    monkeypatch,
    failure_point,
):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    write_csv(
        selection_path,
        SELECTION_FIELDS,
        [selection_row("A", "ground")],
    )
    write_csv(
        candidates_path,
        PACKET_FIELDS,
        [candidate_row("A", "C0001")],
    )
    packet_path.write_bytes(b"existing packet\r\n")
    template_path.write_bytes(b"existing template\r\n")
    packet_path.chmod(0o640)
    template_path.chmod(0o600)
    original_packet = packet_path.read_bytes()
    original_template = template_path.read_bytes()
    original_packet_mode = stat.S_IMODE(packet_path.stat().st_mode)
    original_template_mode = stat.S_IMODE(template_path.stat().st_mode)

    if failure_point == "second_stage":
        real_writerows = coding_reliability.csv.DictWriter.writerows
        calls = 0

        def fail_second_writerows(writer, rows):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second staging failure")
            return real_writerows(writer, rows)

        monkeypatch.setattr(
            coding_reliability.csv.DictWriter,
            "writerows",
            fail_second_writerows,
        )
    else:
        real_exchange = coding_reliability._rename_exchange_at

        def fail_template_publish(directory, source_name, target_name):
            if (
                directory.path / target_name == template_path
                and source_name.startswith(f".{template_path.name}.")
                and source_name.endswith(".tmp")
            ):
                raise OSError("injected second publication failure")
            return real_exchange(
                directory,
                source_name,
                target_name,
            )

        monkeypatch.setattr(
            coding_reliability,
            "_rename_exchange_at",
            fail_template_publish,
        )

    with pytest.raises(OSError, match="injected second"):
        main(
            [
                "--prepare-blind",
                "--selection",
                str(selection_path),
                "--candidates",
                str(candidates_path),
                "--packet-output",
                str(packet_path),
                "--template-output",
                str(template_path),
            ]
        )

    assert packet_path.read_bytes() == original_packet
    assert template_path.read_bytes() == original_template
    assert stat.S_IMODE(packet_path.stat().st_mode) == original_packet_mode
    assert stat.S_IMODE(template_path.stat().st_mode) == original_template_mode
    assert not hidden_output_artifacts(packet_path, template_path)

def test_prepare_blind_rolls_back_when_backup_link_then_raises(
    tmp_path,
    monkeypatch,
):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    write_csv(
        selection_path,
        SELECTION_FIELDS,
        [selection_row("A", "ground")],
    )
    write_csv(
        candidates_path,
        PACKET_FIELDS,
        [candidate_row("A", "C0001")],
    )
    packet_path.write_bytes(b"existing packet\r\n")
    template_path.write_bytes(b"existing template\r\n")
    original_packet = packet_path.read_bytes()
    original_template = template_path.read_bytes()
    real_link = coding_reliability._link_fd_at
    injected = False

    def link_backup_then_fail(source_fd, directory, target_name):
        nonlocal injected
        result = real_link(source_fd, directory, target_name)
        if (
            not injected
            and target_name.startswith(f".{packet_path.name}.")
            and target_name.endswith(".bak")
        ):
            injected = True
            raise OSError("injected post-backup failure")
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_link_fd_at",
        link_backup_then_fail,
    )

    with pytest.raises(OSError, match="injected post-backup failure"):
        main(
            [
                "--prepare-blind",
                "--selection",
                str(selection_path),
                "--candidates",
                str(candidates_path),
                "--packet-output",
                str(packet_path),
                "--template-output",
                str(template_path),
            ]
        )

    assert packet_path.read_bytes() == original_packet
    assert template_path.read_bytes() == original_template
    assert not hidden_output_artifacts(packet_path, template_path)


def test_prepare_blind_rolls_back_when_publication_link_then_raises(
    tmp_path,
    monkeypatch,
):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    write_csv(
        selection_path,
        SELECTION_FIELDS,
        [selection_row("A", "ground")],
    )
    write_csv(
        candidates_path,
        PACKET_FIELDS,
        [candidate_row("A", "C0001")],
    )
    template_path.write_bytes(b"existing template\r\n")
    original_template = template_path.read_bytes()
    real_link = coding_reliability._link_fd_at
    injected = False

    def publish_packet_then_fail(source_fd, directory, target_name):
        nonlocal injected
        result = real_link(source_fd, directory, target_name)
        if (
            not injected
            and directory.path / target_name == packet_path
        ):
            injected = True
            raise OSError("injected post-publication failure")
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_link_fd_at",
        publish_packet_then_fail,
    )

    with pytest.raises(OSError, match="injected post-publication failure"):
        main(
            [
                "--prepare-blind",
                "--selection",
                str(selection_path),
                "--candidates",
                str(candidates_path),
                "--packet-output",
                str(packet_path),
                "--template-output",
                str(template_path),
            ]
        )

    assert not packet_path.exists()
    assert template_path.read_bytes() == original_template
    assert not hidden_output_artifacts(packet_path, template_path)



@pytest.mark.parametrize(
    "arguments",
    [
        ["--select", "--prepare-blind", "--evidence", "INPUT", "--output", "OUTPUT"],
        [
            "--materialize-primary",
            "--prepare",
            "--selection",
            "INPUT",
            "--evidence",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        [
            "--prepare-blind",
            "--selection",
            "INPUT",
            "--candidates",
            "INPUT",
            "--packet-output",
            "PACKET",
        ],
        [
            "--materialize-primary",
            "--selection",
            "INPUT",
            "--evidence",
            "INPUT",
        ],
        [
            "--select",
            "--selection",
            "INPUT",
            "--evidence",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        [
            "--prepare-blind",
            "--selection",
            "INPUT",
            "--candidates",
            "INPUT",
            "--packet-output",
            "PACKET",
            "--template-output",
            "TEMPLATE",
            "--output",
            "OUTPUT",
        ],
        [
            "--materialize-primary",
            "--selection",
            "INPUT",
            "--evidence",
            "INPUT",
            "--candidates",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        [
            "--primary",
            "INPUT",
            "--reliability",
            "INPUT",
            "--selection",
            "INPUT",
            "--output",
            "OUTPUT",
        ],
        [
            "--prepare-blind",
            "--selection",
            "INPUT",
            "--candidates",
            "INPUT",
            "--packet-output",
            "PACKET",
            "--template-output",
            "PACKET",
        ],
    ],
)
def test_new_cli_modes_reject_missing_mixed_and_shared_output_arguments(
    tmp_path,
    arguments,
):
    paths = {
        "INPUT": tmp_path / "input.csv",
        "OUTPUT": tmp_path / "output.csv",
        "PACKET": tmp_path / "packet.csv",
        "TEMPLATE": tmp_path / "template.csv",
    }
    resolved = [str(paths.get(argument, argument)) for argument in arguments]

    with pytest.raises(SystemExit) as exc_info:
        main(resolved)

    assert exc_info.value.code == 2
    assert not paths["OUTPUT"].exists()
    assert not paths["PACKET"].exists()
    assert not paths["TEMPLATE"].exists()
    assert not hidden_output_artifacts(
        paths["OUTPUT"],
        paths["PACKET"],
        paths["TEMPLATE"],
    )


def test_generated_blank_template_is_allowed_then_rejected_by_compare(tmp_path):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "coding_reliability.csv"
    primary_path = tmp_path / "coding_primary.csv"
    summary_path = tmp_path / "summary.csv"
    write_csv(
        selection_path,
        SELECTION_FIELDS,
        [selection_row("A", "ground")],
    )
    write_csv(
        candidates_path,
        PACKET_FIELDS,
        [candidate_row("A", "C0001")],
    )
    write_csv(primary_path, TEMPLATE_FIELDS, [coding_row("A")])

    main(
        [
            "--prepare-blind",
            "--selection",
            str(selection_path),
            "--candidates",
            str(candidates_path),
            "--packet-output",
            str(packet_path),
            "--template-output",
            str(template_path),
        ]
    )

    template_header, template = read_csv(template_path)
    assert template_header == TEMPLATE_FIELDS
    assert template == [
        {"cite_key": "A", **dict.fromkeys(CORE_FIELDS, "")}
    ]

    with pytest.raises(
        ValueError,
        match="reliability.*field.*must have a nonempty canonical value",
    ):
        main(
            [
                "--primary",
                str(primary_path),
                "--reliability",
                str(template_path),
                "--output",
                str(summary_path),
            ]
        )

    assert not summary_path.exists()
    assert not staged_files(summary_path)

@pytest.mark.parametrize(
    "alias_kind",
    ["lexical_parent", "relative_absolute", "symlink", "hardlink"],
)
def test_select_rejects_filesystem_aliases_before_publication(
    tmp_path,
    monkeypatch,
    alias_kind,
):
    evidence_path = tmp_path / "evidence.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    original_evidence = evidence_path.read_bytes()
    evidence_argument = evidence_path

    if alias_kind == "lexical_parent":
        alias_parent = tmp_path / "alias-parent"
        alias_parent.mkdir()
        output_path = alias_parent / ".." / evidence_path.name
    elif alias_kind == "relative_absolute":
        monkeypatch.chdir(tmp_path)
        evidence_argument = Path(evidence_path.name)
        output_path = evidence_path.resolve()
    elif alias_kind == "symlink":
        output_path = tmp_path / "selection.csv"
        output_path.symlink_to(evidence_path)
    else:
        output_path = tmp_path / "selection.csv"
        os.link(evidence_path, output_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--select",
                "--evidence",
                str(evidence_argument),
                "--output",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert evidence_path.read_bytes() == original_evidence
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


@pytest.mark.parametrize(
    ("mode", "left", "right"),
    [
        ("prepare", "evidence", "output"),
        ("select", "evidence", "output"),
        ("prepare_blind", "selection", "candidates"),
        ("prepare_blind", "selection", "packet_output"),
        ("prepare_blind", "selection", "template_output"),
        ("prepare_blind", "candidates", "packet_output"),
        ("prepare_blind", "candidates", "template_output"),
        ("prepare_blind", "packet_output", "template_output"),
        ("materialize_primary", "selection", "evidence"),
        ("materialize_primary", "selection", "output"),
        ("materialize_primary", "evidence", "output"),
        ("compare", "primary", "output"),
        ("compare", "reliability", "output"),
    ],
)
def test_every_mode_rejects_each_aliased_path_pair_before_reading(
    tmp_path,
    monkeypatch,
    mode,
    left,
    right,
):
    mode_contracts = {
        "prepare": (["--prepare"], ("evidence", "output")),
        "select": (["--select"], ("evidence", "output")),
        "prepare_blind": (
            ["--prepare-blind"],
            ("selection", "candidates", "packet_output", "template_output"),
        ),
        "materialize_primary": (
            ["--materialize-primary"],
            ("selection", "evidence", "output"),
        ),
        "compare": ([], ("primary", "reliability", "output")),
    }
    flags, names = mode_contracts[mode]
    paths = {name: tmp_path / f"{name}.csv" for name in names}
    paths[right] = paths[left]
    arguments = list(flags)
    for name in names:
        arguments.extend(
            [
                f"--{name.replace('_', '-')}",
                str(paths[name]),
            ]
        )

    def fail_if_read(_path):
        raise AssertionError("aliased paths reached input reading")

    monkeypatch.setattr(
        coding_reliability,
        "_read_csv_with_identity",
        fail_if_read,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(arguments)

    assert exc_info.value.code == 2
    assert not list(tmp_path.iterdir())


def test_compare_rejects_same_primary_and_reliability_as_independence_guard(
    tmp_path,
    capsys,
):
    coding_path = tmp_path / "coding.csv"
    output_path = tmp_path / "summary.csv"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--primary",
                str(coding_path),
                "--reliability",
                str(coding_path),
                "--output",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert "--primary and --reliability must reference distinct paths" in (
        capsys.readouterr().err
    )
    assert not output_path.exists()


def test_compare_rejects_inputs_hardlinked_between_captures(
    tmp_path,
    monkeypatch,
):
    primary_path = tmp_path / "primary.csv"
    reliability_path = tmp_path / "reliability.csv"
    output_path = tmp_path / "summary.csv"
    rows = [coding_row("A")]
    write_csv(primary_path, TEMPLATE_FIELDS, rows)
    write_csv(reliability_path, TEMPLATE_FIELDS, rows)

    primary_identity = primary_path.stat().st_ino
    reliability_identity = reliability_path.stat().st_ino
    assert primary_identity != reliability_identity

    real_read = coding_reliability._read_csv_with_identity
    swapped = False

    def read_then_alias(path):
        nonlocal swapped
        result = real_read(path)
        if path == primary_path and not swapped:
            swapped = True
            reliability_path.unlink()
            os.link(primary_path, reliability_path)
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_read_csv_with_identity",
        read_then_alias,
    )

    with pytest.raises(ValueError, match="independent.*same file"):
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

    assert swapped
    assert primary_path.stat().st_ino == reliability_path.stat().st_ino
    assert not output_path.exists()
    assert not hidden_output_artifacts(output_path)


def test_compare_reattests_both_paths_after_capture_before_comparison(
    tmp_path,
    monkeypatch,
):
    primary_path = tmp_path / "primary.csv"
    reliability_path = tmp_path / "reliability.csv"
    output_path = tmp_path / "summary.csv"
    rows = [coding_row("A")]
    write_csv(primary_path, TEMPLATE_FIELDS, rows)
    write_csv(reliability_path, TEMPLATE_FIELDS, rows)

    real_read = coding_reliability._read_csv_with_identity
    swapped = False

    def read_then_retarget_primary(path):
        nonlocal swapped
        result = real_read(path)
        if path == reliability_path and not swapped:
            swapped = True
            primary_path.unlink()
            os.link(reliability_path, primary_path)
        return result

    def fail_if_compared(_primary, _reliability):
        raise AssertionError("changed captures reached comparison")

    monkeypatch.setattr(
        coding_reliability,
        "_read_csv_with_identity",
        read_then_retarget_primary,
    )
    monkeypatch.setattr(
        coding_reliability,
        "compare_codings",
        fail_if_compared,
    )

    with pytest.raises(ValueError, match="input.*changed"):
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

    assert swapped
    assert primary_path.stat().st_ino == reliability_path.stat().st_ino
    assert not output_path.exists()
    assert not hidden_output_artifacts(output_path)


@pytest.mark.parametrize("single_mode", ["select", "materialize_primary"])
def test_single_output_post_publication_failure_restores_original(
    tmp_path,
    monkeypatch,
    single_mode,
):
    evidence_path = tmp_path / "evidence.csv"
    selection_path = tmp_path / "selection.csv"
    output_path = tmp_path / "output.csv"

    if single_mode == "select":
        write_csv(
            evidence_path,
            ("cite_key", "domain"),
            [{"cite_key": "A", "domain": "ground"}],
        )
        arguments = [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    else:
        header = ("cite_key", "survey_evidence_tier", "domain", *CORE_FIELDS[1:], "title", "coding_notes")
        evidence = [
            complete_evidence_row("A", "ground"),
            complete_evidence_row("B", "ground"),
        ]
        write_csv(evidence_path, header, evidence)
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(selection_path),
            ]
        )
        arguments = [
            "--materialize-primary",
            "--selection",
            str(selection_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]

    output_path.write_bytes(b"existing single output\r\n")
    output_path.chmod(0o640)
    original_bytes = output_path.read_bytes()
    original_mode = stat.S_IMODE(output_path.stat().st_mode)
    real_exchange = coding_reliability._rename_exchange_at
    injected = False

    def publish_then_fail(directory, source_name, target_name):
        nonlocal injected
        result = real_exchange(directory, source_name, target_name)
        if (
            not injected
            and directory.path / target_name == output_path
            and source_name.startswith(f".{output_path.name}.")
            and source_name.endswith(".tmp")
        ):
            injected = True
            raise OSError("injected single post-publication failure")
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_rename_exchange_at",
        publish_then_fail,
    )

    with pytest.raises(
        OSError,
        match="injected single post-publication failure",
    ):
        main(arguments)

    assert output_path.read_bytes() == original_bytes
    assert stat.S_IMODE(output_path.stat().st_mode) == original_mode
    assert not hidden_output_artifacts(output_path)


def test_parent_retarget_after_path_validation_cannot_overwrite_input(
    tmp_path,
    monkeypatch,
):
    input_parent = tmp_path / "inputs"
    output_parent = tmp_path / "outputs"
    displaced_parent = tmp_path / "outputs-original"
    input_parent.mkdir()
    output_parent.mkdir()
    evidence_path = input_parent / "shared.csv"
    output_path = output_parent / "shared.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    original_evidence = evidence_path.read_bytes()
    real_validate = coding_reliability._validate_distinct_paths

    def validate_then_retarget(parser, paths):
        real_validate(parser, paths)
        output_parent.rename(displaced_parent)
        output_parent.symlink_to(input_parent, target_is_directory=True)

    monkeypatch.setattr(
        coding_reliability,
        "_validate_distinct_paths",
        validate_then_retarget,
    )

    with pytest.raises((ValueError, OSError, SystemExit)):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert evidence_path.read_bytes() == original_evidence
    assert not (displaced_parent / output_path.name).exists()
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


def test_parent_retarget_during_staging_cleans_anchored_artifacts(
    tmp_path,
    monkeypatch,
):
    input_parent = tmp_path / "inputs"
    output_parent = tmp_path / "outputs"
    displaced_parent = tmp_path / "outputs-original"
    input_parent.mkdir()
    output_parent.mkdir()
    evidence_path = input_parent / "shared.csv"
    output_path = output_parent / "shared.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    original_evidence = evidence_path.read_bytes()
    real_writerows = coding_reliability.csv.DictWriter.writerows
    injected = False

    def write_then_retarget(writer, rows):
        nonlocal injected
        result = real_writerows(writer, rows)
        if not injected:
            injected = True
            output_parent.rename(displaced_parent)
            output_parent.symlink_to(input_parent, target_is_directory=True)
        return result

    monkeypatch.setattr(
        coding_reliability.csv.DictWriter,
        "writerows",
        write_then_retarget,
    )

    with pytest.raises((ValueError, OSError)):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert evidence_path.read_bytes() == original_evidence
    assert not (displaced_parent / output_path.name).exists()
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


@pytest.mark.parametrize("input_drift", ["bytes", "identity"])
def test_input_is_revalidated_immediately_before_single_output_publish(
    tmp_path,
    monkeypatch,
    input_drift,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain", "notes"),
        [{"cite_key": "A", "domain": "ground", "notes": "original"}],
    )
    original_evidence = evidence_path.read_bytes()
    output_path.write_bytes(b"existing selection\r\n")
    original_output = output_path.read_bytes()
    real_writerows = coding_reliability.csv.DictWriter.writerows
    injected = False

    def write_then_change_input(writer, rows):
        nonlocal injected
        result = real_writerows(writer, rows)
        if not injected:
            injected = True
            if input_drift == "bytes":
                evidence_path.write_bytes(original_evidence + b"\n")
            else:
                replacement = tmp_path / "replacement.csv"
                replacement.write_bytes(original_evidence)
                replacement.replace(evidence_path)
        return result

    monkeypatch.setattr(
        coding_reliability.csv.DictWriter,
        "writerows",
        write_then_change_input,
    )

    with pytest.raises(ValueError, match="input.*changed"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not hidden_output_artifacts(output_path)


@pytest.mark.parametrize(
    "output_kind",
    ["directory", "symlink", "fifo", "device"],
)
def test_existing_nonregular_output_is_rejected_before_staging(
    tmp_path,
    monkeypatch,
    output_kind,
):
    evidence_path = tmp_path / "evidence.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    sentinel_path = tmp_path / "sentinel.txt"
    sentinel_path.write_bytes(b"sentinel\n")

    if output_kind == "device":
        output_path = Path("/dev/null")
    else:
        output_path = tmp_path / "output.csv"
        if output_kind == "directory":
            output_path.mkdir()
        elif output_kind == "symlink":
            output_path.symlink_to(sentinel_path)
        else:
            os.mkfifo(output_path)

    def fail_if_staged(_writer, _rows):
        raise AssertionError("non-regular output reached CSV staging")

    monkeypatch.setattr(
        coding_reliability.csv.DictWriter,
        "writerows",
        fail_if_staged,
    )

    with pytest.raises(ValueError, match="regular"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    if output_kind == "directory":
        assert output_path.is_dir()
    elif output_kind == "symlink":
        assert output_path.is_symlink()
        assert output_path.readlink() == sentinel_path
        assert sentinel_path.read_bytes() == b"sentinel\n"
    elif output_kind == "fifo":
        assert stat.S_ISFIFO(output_path.lstat().st_mode)
    else:
        assert stat.S_ISCHR(output_path.stat().st_mode)
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


def test_post_backup_cleanup_failure_rolls_back_all_outputs(
    tmp_path,
    monkeypatch,
):
    selection_path = tmp_path / "selection.csv"
    candidates_path = tmp_path / "candidates.csv"
    packet_path = tmp_path / "packet.csv"
    template_path = tmp_path / "template.csv"
    write_csv(
        selection_path,
        SELECTION_FIELDS,
        [selection_row("A", "ground")],
    )
    write_csv(
        candidates_path,
        PACKET_FIELDS,
        [candidate_row("A", "C0001")],
    )
    packet_path.write_bytes(b"existing packet\r\n")
    template_path.write_bytes(b"existing template\r\n")
    packet_path.chmod(0o640)
    template_path.chmod(0o600)
    original_packet = packet_path.read_bytes()
    original_template = template_path.read_bytes()
    original_packet_mode = stat.S_IMODE(packet_path.stat().st_mode)
    original_template_mode = stat.S_IMODE(template_path.stat().st_mode)
    real_capture = coding_reliability._capture_then_classify_entry
    injected = False

    def capture_backup_then_fail(state, name, expected_identity):
        nonlocal injected
        result = real_capture(state, name, expected_identity)
        if not injected and name.endswith(".bak"):
            injected = True
            raise OSError("injected post-cleanup failure")
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_capture_then_classify_entry",
        capture_backup_then_fail,
    )

    with pytest.raises(OSError, match="injected post-cleanup failure"):
        main(
            [
                "--prepare-blind",
                "--selection",
                str(selection_path),
                "--candidates",
                str(candidates_path),
                "--packet-output",
                str(packet_path),
                "--template-output",
                str(template_path),
            ]
        )

    assert packet_path.read_bytes() == original_packet
    assert template_path.read_bytes() == original_template
    assert stat.S_IMODE(packet_path.stat().st_mode) == original_packet_mode
    assert stat.S_IMODE(template_path.stat().st_mode) == original_template_mode
    assert not hidden_output_artifacts(packet_path, template_path)



def test_final_precleanup_parent_retarget_rolls_back_installed_output(
    tmp_path,
    monkeypatch,
):
    output_parent = tmp_path / "outputs"
    displaced_parent = tmp_path / "outputs-original"
    output_parent.mkdir()
    evidence_path = tmp_path / "evidence.csv"
    output_path = output_parent / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    original_evidence = evidence_path.read_bytes()
    real_link = coding_reliability._link_fd_at
    injected = False

    def publish_then_retarget(source_fd, directory, target_name):
        nonlocal injected
        result = real_link(source_fd, directory, target_name)
        if (
            not injected
            and directory.path / target_name == output_path
        ):
            injected = True
            output_parent.rename(displaced_parent)
            output_parent.mkdir()
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_link_fd_at",
        publish_then_retarget,
    )

    with pytest.raises(ValueError, match="output parent changed"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert evidence_path.read_bytes() == original_evidence
    assert not output_path.exists()
    assert not (displaced_parent / output_path.name).exists()
    assert not list(tmp_path.rglob(".*.tmp"))
    assert not list(tmp_path.rglob(".*.bak"))


def test_final_precleanup_evidence_mutation_restores_original_output(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain", "notes"),
        [{"cite_key": "A", "domain": "ground", "notes": "original"}],
    )
    original_evidence = evidence_path.read_bytes()
    output_path.write_bytes(b"existing selection\r\n")
    output_path.chmod(0o640)
    original_output = output_path.read_bytes()
    original_mode = stat.S_IMODE(output_path.stat().st_mode)
    real_exchange = coding_reliability._rename_exchange_at
    injected = False

    def publish_then_mutate_evidence(directory, source_name, target_name):
        nonlocal injected
        result = real_exchange(directory, source_name, target_name)
        if (
            not injected
            and directory.path / target_name == output_path
            and source_name.endswith(".tmp")
        ):
            injected = True
            evidence_path.write_bytes(original_evidence + b"\n")
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_rename_exchange_at",
        publish_then_mutate_evidence,
    )

    with pytest.raises(ValueError, match="input.*changed"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert stat.S_IMODE(output_path.stat().st_mode) == original_mode
    assert not hidden_output_artifacts(output_path)


@pytest.mark.parametrize("existing_output", [False, True])
def test_final_install_refuses_concurrent_target_creation_or_swap(
    tmp_path,
    monkeypatch,
    existing_output,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    if existing_output:
        output_path.write_bytes(b"original output\n")

    concurrent_bytes = b"concurrent target\n"
    concurrent_identity = None
    real_revalidate_input = coding_reliability._revalidate_input
    injected = False

    def revalidate_then_race(snapshot):
        nonlocal concurrent_identity, injected
        result = real_revalidate_input(snapshot)
        if not injected:
            injected = True
            replacement = tmp_path / "concurrent.csv"
            replacement.write_bytes(concurrent_bytes)
            if existing_output:
                replacement.replace(output_path)
            else:
                os.link(replacement, output_path)
            status = output_path.stat()
            concurrent_identity = (status.st_dev, status.st_ino)
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_input",
        revalidate_then_race,
    )

    with pytest.raises((OSError, RuntimeError, ValueError)):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    status = output_path.stat()
    assert output_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == concurrent_identity


def test_new_output_swap_after_no_replace_link_is_reported_as_conflict(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    concurrent_bytes = b"concurrent post-link target\n"
    concurrent_identity = None
    real_link = coding_reliability._link_fd_at
    injected = False

    def link_then_swap(source_fd, directory, target_name):
        nonlocal concurrent_identity, injected
        result = real_link(source_fd, directory, target_name)
        if (
            not injected
            and directory.path / target_name == output_path
        ):
            injected = True
            replacement = tmp_path / "concurrent-post-link.csv"
            replacement.write_bytes(concurrent_bytes)
            replacement.replace(output_path)
            status = output_path.stat()
            concurrent_identity = (status.st_dev, status.st_ino)
        return result

    monkeypatch.setattr(
        coding_reliability,
        "_link_fd_at",
        link_then_swap,
    )

    with pytest.raises(RuntimeError, match="installation.*raced"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    status = output_path.stat()
    assert output_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == concurrent_identity


def test_backup_creation_never_replaces_concurrent_backup_entry(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"original output\n")
    original_identity = output_path.stat().st_ino
    concurrent_bytes = b"concurrent backup\n"
    backup_path = None
    backup_identity = None
    real_unused_name = coding_reliability._unused_entry_name

    def allocate_then_create_backup(state, suffix):
        nonlocal backup_identity, backup_path
        name = real_unused_name(state, suffix)
        if suffix == "bak" and backup_path is None:
            backup_path = state.directory.path / name
            backup_path.write_bytes(concurrent_bytes)
            status = backup_path.stat()
            backup_identity = (status.st_dev, status.st_ino)
        return name

    monkeypatch.setattr(
        coding_reliability,
        "_unused_entry_name",
        allocate_then_create_backup,
    )

    with pytest.raises(OSError):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == b"original output\n"
    assert output_path.stat().st_ino == original_identity
    assert backup_path is not None
    status = backup_path.stat()
    assert backup_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == backup_identity


def test_cleanup_captures_expected_transaction_without_deletion(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    cleanup_destinations = []
    rename_noreplace = coding_reliability._rename_noreplace_at

    def record_cleanup_destination(directory, source_name, destination_name):
        if (
            source_name.endswith(".tmp")
            and destination_name.startswith(".trackgen-retired-")
        ):
            cleanup_destinations.append(destination_name)
        return rename_noreplace(directory, source_name, destination_name)

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        record_cleanup_destination,
    )
    monkeypatch.setattr(coding_reliability.os, "unlink", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "remove", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "rmdir", forbidden_delete)

    main(
        [
            "--select",
            "--evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    )

    assert len(cleanup_destinations) == 1
    assert cleanup_destinations[0].startswith(".trackgen-retired-")
    retired = list(tmp_path.glob(".trackgen-retired-*"))
    assert len(retired) == 1
    assert retired[0].read_bytes() == output_path.read_bytes()
    assert (retired[0].stat().st_dev, retired[0].stat().st_ino) == (
        output_path.stat().st_dev,
        output_path.stat().st_ino,
    )


def test_cleanup_quarantine_replacement_is_restored_without_deletion(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    replacement = tmp_path / "concurrent-cleanup.csv"
    replacement.write_bytes(b"concurrent cleanup entry\n")
    replacement_identity = (
        replacement.stat().st_dev,
        replacement.stat().st_ino,
    )
    parked_owned = tmp_path / "parked-owned-output.csv"
    concurrent_path = None
    owned_identity = None
    injected = False
    rename_noreplace = coding_reliability._rename_noreplace_at

    def rename_then_swap(directory, source_name, destination_name):
        nonlocal concurrent_path, injected, owned_identity
        rename_noreplace(directory, source_name, destination_name)
        if (
            not injected
            and source_name.endswith(".tmp")
            and destination_name.startswith(".trackgen-retired-")
        ):
            injected = True
            concurrent_path = directory.path / source_name
            retired_path = directory.path / destination_name
            status = retired_path.stat()
            owned_identity = (status.st_dev, status.st_ino)
            retired_path.rename(parked_owned)
            replacement.rename(retired_path)

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        rename_then_swap,
    )
    monkeypatch.setattr(coding_reliability.os, "unlink", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "remove", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "rmdir", forbidden_delete)

    with pytest.raises(RuntimeError, match="captured foreign entry"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert injected
    assert not output_path.exists()
    assert concurrent_path is not None
    assert concurrent_path.read_bytes() == b"concurrent cleanup entry\n"
    assert (concurrent_path.stat().st_dev, concurrent_path.stat().st_ino) == (
        replacement_identity
    )
    assert owned_identity is not None
    assert (parked_owned.stat().st_dev, parked_owned.stat().st_ino) == (
        owned_identity
    )
    assert any(
        (path.stat().st_dev, path.stat().st_ino) == owned_identity
        for path in tmp_path.glob(".trackgen-retired-*")
    )


def test_cleanup_capture_then_classify_retains_foreign_file_when_source_refills(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    captured_foreign = tmp_path / "captured-foreign.csv"
    captured_foreign.write_bytes(b"captured foreign cleanup entry\n")
    captured_identity = (
        captured_foreign.stat().st_dev,
        captured_foreign.stat().st_ino,
    )
    refill_foreign = tmp_path / "refill-foreign.csv"
    refill_foreign.write_bytes(b"refilled foreign cleanup entry\n")
    refill_identity = (
        refill_foreign.stat().st_dev,
        refill_foreign.stat().st_ino,
    )
    parked_expected = tmp_path / "parked-expected-stage.csv"
    cleanup_started = False
    source_replaced = False
    source_refilled = False
    source_path = None
    quarantine_path = None
    real_cleanup = coding_reliability._cleanup_committed_artifacts
    real_entry_status = coding_reliability._entry_status
    real_rename = coding_reliability._rename_noreplace_at

    def start_cleanup(state):
        nonlocal cleanup_started
        cleanup_started = True
        return real_cleanup(state)

    def status_then_replace(directory, name):
        nonlocal source_path, source_replaced
        status = real_entry_status(directory, name)
        if (
            cleanup_started
            and not source_replaced
            and name.endswith(".tmp")
            and status is not None
        ):
            source_path = directory.path / name
            source_path.rename(parked_expected)
            captured_foreign.rename(source_path)
            source_replaced = True
        return status

    def capture_then_refill(directory, source_name, destination_name):
        nonlocal quarantine_path, source_refilled
        result = real_rename(directory, source_name, destination_name)
        if (
            source_replaced
            and not source_refilled
            and source_name.endswith(".tmp")
            and destination_name.startswith(".trackgen-retired-")
        ):
            quarantine_path = directory.path / destination_name
            os.link(refill_foreign, directory.path / source_name)
            source_refilled = True
        return result

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(
        coding_reliability,
        "_cleanup_committed_artifacts",
        start_cleanup,
    )
    monkeypatch.setattr(coding_reliability, "_entry_status", status_then_replace)
    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        capture_then_refill,
    )
    monkeypatch.setattr(coding_reliability.os, "unlink", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "remove", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "rmdir", forbidden_delete)

    with pytest.raises(Exception) as raised:
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert source_replaced
    assert source_refilled
    assert source_path is not None
    assert quarantine_path is not None
    assert quarantine_path.is_absolute()
    assert quarantine_path.read_bytes() == b"captured foreign cleanup entry\n"
    assert (quarantine_path.stat().st_dev, quarantine_path.stat().st_ino) == (
        captured_identity
    )
    assert source_path.read_bytes() == b"refilled foreign cleanup entry\n"
    assert (source_path.stat().st_dev, source_path.stat().st_ino) == refill_identity
    recovery = _exception_text(raised.value)
    assert str(quarantine_path) in recovery
    assert f"(dev, ino)=({captured_identity[0]}, {captured_identity[1]})" in recovery


def test_backup_boundary_detects_target_swap_without_losing_replacement(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"original output\n")
    concurrent_bytes = b"concurrent target\n"
    concurrent_identity = None
    real_unused_name = coding_reliability._unused_entry_name
    injected = False

    def allocate_then_swap_target(state, suffix):
        nonlocal concurrent_identity, injected
        name = real_unused_name(state, suffix)
        if suffix == "bak" and not injected:
            injected = True
            replacement = tmp_path / "concurrent.csv"
            replacement.write_bytes(concurrent_bytes)
            replacement.replace(output_path)
            status = output_path.stat()
            concurrent_identity = (status.st_dev, status.st_ino)
        return name

    monkeypatch.setattr(
        coding_reliability,
        "_unused_entry_name",
        allocate_then_swap_target,
    )

    with pytest.raises((OSError, RuntimeError, ValueError)):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    status = output_path.stat()
    assert output_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == concurrent_identity


@pytest.mark.parametrize("existing_output", [False, True])
def test_rollback_precheck_replacement_is_compensated_without_overwrite(
    tmp_path,
    monkeypatch,
    existing_output,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    if existing_output:
        output_path.write_bytes(b"original output\n")

    concurrent_bytes = b"concurrent rollback target\n"
    concurrent_identity = None
    rollback_started = False
    injected = False
    real_entry_status = coding_reliability._entry_status

    def fail_after_install(_state):
        nonlocal rollback_started
        rollback_started = True
        raise OSError("primary publication failure")

    def status_then_swap(directory, name):
        nonlocal concurrent_identity, injected
        status = real_entry_status(directory, name)
        if (
            rollback_started
            and not injected
            and directory.path / name == output_path
        ):
            injected = True
            replacement = tmp_path / "concurrent-rollback.csv"
            replacement.write_bytes(concurrent_bytes)
            replacement.replace(output_path)
            replacement_status = output_path.stat()
            concurrent_identity = (
                replacement_status.st_dev,
                replacement_status.st_ino,
            )
        return status

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_installed_output",
        fail_after_install,
    )
    monkeypatch.setattr(
        coding_reliability,
        "_entry_status",
        status_then_swap,
    )

    with pytest.raises(OSError, match="primary publication failure"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    status = output_path.stat()
    assert output_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == concurrent_identity


def test_new_output_rollback_quarantines_foreign_file_when_source_refills(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    captured_foreign = tmp_path / "captured-rollback-foreign.csv"
    captured_foreign.write_bytes(b"captured foreign rollback output\n")
    captured_identity = (
        captured_foreign.stat().st_dev,
        captured_foreign.stat().st_ino,
    )
    refill_foreign = tmp_path / "refill-rollback-foreign.csv"
    refill_foreign.write_bytes(b"refilled foreign rollback output\n")
    refill_identity = (
        refill_foreign.stat().st_dev,
        refill_foreign.stat().st_ino,
    )
    parked_expected = tmp_path / "parked-published-output.csv"
    rollback_started = False
    source_replaced = False
    source_refilled = False
    quarantine_path = None
    real_entry_status = coding_reliability._entry_status
    real_rename = coding_reliability._rename_noreplace_at

    def fail_after_install(_state):
        nonlocal rollback_started
        rollback_started = True
        raise OSError("primary publication failure")

    def status_then_replace(directory, name):
        nonlocal source_replaced
        status = real_entry_status(directory, name)
        if (
            rollback_started
            and not source_replaced
            and directory.path / name == output_path
            and status is not None
        ):
            output_path.rename(parked_expected)
            captured_foreign.rename(output_path)
            source_replaced = True
        return status

    def capture_then_refill(directory, source_name, destination_name):
        nonlocal quarantine_path, source_refilled
        result = real_rename(directory, source_name, destination_name)
        if (
            source_replaced
            and not source_refilled
            and directory.path / source_name == output_path
            and destination_name.startswith(".trackgen-retired-")
        ):
            quarantine_path = directory.path / destination_name
            os.link(refill_foreign, output_path)
            source_refilled = True
        return result

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("rollback must not delete a mutable pathname")

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_installed_output",
        fail_after_install,
    )
    monkeypatch.setattr(coding_reliability, "_entry_status", status_then_replace)
    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        capture_then_refill,
    )
    monkeypatch.setattr(coding_reliability.os, "unlink", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "remove", forbidden_delete)
    monkeypatch.setattr(coding_reliability.os, "rmdir", forbidden_delete)

    with pytest.raises(OSError, match="primary publication failure") as raised:
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert source_replaced
    assert source_refilled
    assert quarantine_path is not None
    assert quarantine_path.is_absolute()
    assert quarantine_path.read_bytes() == b"captured foreign rollback output\n"
    assert (quarantine_path.stat().st_dev, quarantine_path.stat().st_ino) == (
        captured_identity
    )
    assert output_path.read_bytes() == b"refilled foreign rollback output\n"
    assert (output_path.stat().st_dev, output_path.stat().st_ino) == refill_identity
    recovery = _exception_text(raised.value)
    assert str(quarantine_path) in recovery
    assert f"(dev, ino)=({captured_identity[0]}, {captured_identity[1]})" in recovery


def test_rollback_backup_precheck_replacement_is_preserved_on_conflict(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"original output\n")
    concurrent_bytes = b"concurrent rollback backup\n"
    concurrent_identity = None
    concurrent_backup_path = None
    rollback_started = False
    injected = False
    real_entry_status = coding_reliability._entry_status

    def fail_after_install(_state):
        nonlocal rollback_started
        rollback_started = True
        raise OSError("primary publication failure")

    def status_then_swap_backup(directory, name):
        nonlocal concurrent_backup_path, concurrent_identity, injected
        status = real_entry_status(directory, name)
        if rollback_started and not injected and name.endswith(".bak"):
            injected = True
            concurrent_backup_path = directory.path / name
            replacement = tmp_path / "concurrent-backup.csv"
            replacement.write_bytes(concurrent_bytes)
            replacement.replace(concurrent_backup_path)
            replacement_status = concurrent_backup_path.stat()
            concurrent_identity = (
                replacement_status.st_dev,
                replacement_status.st_ino,
            )
        return status

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_installed_output",
        fail_after_install,
    )
    monkeypatch.setattr(
        coding_reliability,
        "_entry_status",
        status_then_swap_backup,
    )

    with pytest.raises(OSError, match="primary publication failure"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert concurrent_backup_path is not None
    status = concurrent_backup_path.stat()
    assert concurrent_backup_path.read_bytes() == concurrent_bytes
    assert (status.st_dev, status.st_ino) == concurrent_identity


def test_rollback_conflict_preserves_primary_publication_error(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    concurrent_bytes = b"concurrent rollback target\n"

    def replace_target_then_fail(_state):
        replacement = tmp_path / "concurrent.csv"
        replacement.write_bytes(concurrent_bytes)
        replacement.replace(output_path)
        raise OSError("primary publication failure")

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_installed_output",
        replace_target_then_fail,
    )

    with pytest.raises(OSError, match="primary publication failure"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == concurrent_bytes


def test_post_create_preidentity_failure_cleans_staged_file(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"existing selection\r\n")
    original_output = output_path.read_bytes()
    real_fstat = coding_reliability.os.fstat
    injected = False

    def fail_first_staged_fstat(file_descriptor):
        nonlocal injected
        descriptor_path = os.readlink(
            f"/proc/self/fd/{file_descriptor}"
        )
        if (
            not injected
            and Path(descriptor_path).name.startswith(
                f".{output_path.name}."
            )
            and descriptor_path.endswith(".tmp")
        ):
            injected = True
            raise OSError("injected post-create preidentity failure")
        return real_fstat(file_descriptor)

    monkeypatch.setattr(
        coding_reliability.os,
        "fstat",
        fail_first_staged_fstat,
    )

    with pytest.raises(
        OSError,
        match="injected post-create preidentity failure",
    ):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_bytes() == original_output
    assert not hidden_output_artifacts(output_path)


def _direct_output_state(path: Path):
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    directory_status = os.fstat(directory_fd)
    directory = coding_reliability._DirectoryAnchor(
        path=path.parent.absolute(),
        fd=directory_fd,
        device=directory_status.st_dev,
        inode=directory_status.st_ino,
    )
    return coding_reliability._OutputState(
        path=path.absolute(),
        directory=directory,
        name=path.name,
        rows=[],
        header=(),
        original_bytes=None,
        original_mode=None,
        original_identity=None,
        original_fd=None,
    )


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize(
    "error_number",
    [errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL],
)
def test_rename_noreplace_preflight_fails_before_public_installation(
    tmp_path,
    monkeypatch,
    existing_output,
    error_number,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    original_identity = None
    if existing_output:
        output_path.write_bytes(b"original output\n")
        original_identity = (output_path.stat().st_dev, output_path.stat().st_ino)

    real_noreplace = coding_reliability._rename_noreplace_at
    real_link = coding_reliability._link_fd_at
    real_exchange = coding_reliability._rename_exchange_at
    public_mutation_attempted = False

    def unsupported_probe(directory, source_name, target_name):
        if source_name == target_name:
            raise OSError(error_number, "injected unsupported RENAME_NOREPLACE")
        return real_noreplace(directory, source_name, target_name)

    def record_public_link(source_fd, directory, target_name):
        nonlocal public_mutation_attempted
        if directory.path / target_name == output_path:
            public_mutation_attempted = True
        return real_link(source_fd, directory, target_name)

    def record_public_exchange(directory, source_name, target_name):
        nonlocal public_mutation_attempted
        if directory.path / target_name == output_path:
            public_mutation_attempted = True
        return real_exchange(directory, source_name, target_name)

    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        unsupported_probe,
    )
    monkeypatch.setattr(coding_reliability, "_link_fd_at", record_public_link)
    monkeypatch.setattr(
        coding_reliability,
        "_rename_exchange_at",
        record_public_exchange,
    )

    with pytest.raises(RuntimeError, match="RENAME_NOREPLACE.*output filesystem"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert not public_mutation_attempted
    if existing_output:
        assert output_path.read_bytes() == b"original output\n"
        assert (output_path.stat().st_dev, output_path.stat().st_ino) == (
            original_identity
        )
    else:
        assert not output_path.exists()


def test_new_output_observation_failure_rolls_back_successful_link(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    real_link = coding_reliability._link_fd_at
    real_has_identity = coding_reliability._entry_has_identity
    published = False
    injected = False

    def publish_then_arm(source_fd, directory, target_name):
        nonlocal published
        result = real_link(source_fd, directory, target_name)
        if directory.path / target_name == output_path:
            published = True
        return result

    def fail_first_public_observation(state, name, expected_identity):
        nonlocal injected
        if published and not injected and state.directory.path / name == output_path:
            injected = True
            raise OSError("injected post-link observation failure")
        return real_has_identity(state, name, expected_identity)

    monkeypatch.setattr(coding_reliability, "_link_fd_at", publish_then_arm)
    monkeypatch.setattr(
        coding_reliability,
        "_entry_has_identity",
        fail_first_public_observation,
    )

    with pytest.raises(OSError, match="post-link observation failure"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    assert published
    assert injected
    assert not output_path.exists()


def test_exchange_observation_failure_restores_exact_original_inode(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"original output\n")
    output_path.chmod(0o640)
    os.utime(output_path, ns=(1_600_000_000_000_000_000,) * 2)
    original = output_path.stat()
    real_exchange = coding_reliability._rename_exchange_at
    real_identity_at = coding_reliability._entry_identity_at
    exchanged = False
    injected = False

    def exchange_then_arm(directory, source_name, target_name):
        nonlocal exchanged
        result = real_exchange(directory, source_name, target_name)
        if directory.path / target_name == output_path:
            exchanged = True
        return result

    def fail_first_post_exchange_observation(directory, name):
        nonlocal injected
        if exchanged and not injected:
            injected = True
            raise OSError("injected post-exchange observation failure")
        return real_identity_at(directory, name)

    monkeypatch.setattr(
        coding_reliability,
        "_rename_exchange_at",
        exchange_then_arm,
    )
    monkeypatch.setattr(
        coding_reliability,
        "_entry_identity_at",
        fail_first_post_exchange_observation,
    )

    with pytest.raises(OSError, match="post-exchange observation failure"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    restored = output_path.stat()
    assert exchanged
    assert injected
    assert output_path.read_bytes() == b"original output\n"
    assert (restored.st_dev, restored.st_ino) == (original.st_dev, original.st_ino)
    assert restored.st_mtime_ns == original.st_mtime_ns
    assert stat.S_IMODE(restored.st_mode) == stat.S_IMODE(original.st_mode)


def test_existing_output_rollback_relinks_original_fd_when_names_are_moved(
    tmp_path,
    monkeypatch,
):
    evidence_path = tmp_path / "evidence.csv"
    output_path = tmp_path / "selection.csv"
    parked_backup = tmp_path / "parked-original-backup.csv"
    parked_stage = tmp_path / "parked-original-stage.csv"
    write_csv(
        evidence_path,
        ("cite_key", "domain"),
        [{"cite_key": "A", "domain": "ground"}],
    )
    output_path.write_bytes(b"original output\n")
    output_path.chmod(0o640)
    os.utime(output_path, ns=(1_610_000_000_000_000_000,) * 2)
    original = output_path.stat()

    def move_original_names_then_fail(state):
        assert state.backup_name is not None
        assert state.staged_name is not None
        (state.directory.path / state.backup_name).rename(parked_backup)
        (state.directory.path / state.staged_name).rename(parked_stage)
        raise OSError("injected rollback after original names moved")

    monkeypatch.setattr(
        coding_reliability,
        "_revalidate_installed_output",
        move_original_names_then_fail,
    )

    with pytest.raises(OSError, match="original names moved"):
        main(
            [
                "--select",
                "--evidence",
                str(evidence_path),
                "--output",
                str(output_path),
            ]
        )

    restored = output_path.stat()
    assert output_path.read_bytes() == b"original output\n"
    assert (restored.st_dev, restored.st_ino) == (original.st_dev, original.st_ino)
    assert restored.st_mtime_ns == original.st_mtime_ns
    assert restored.st_uid == original.st_uid
    assert restored.st_gid == original.st_gid
    assert stat.S_IMODE(restored.st_mode) == stat.S_IMODE(original.st_mode)
    assert os.path.samefile(output_path, parked_backup)
    assert os.path.samefile(output_path, parked_stage)


def test_cleanup_bookkeeping_tracks_expected_quarantine_name(tmp_path):
    output_path = tmp_path / "output.csv"
    staged_path = tmp_path / ".output.csv.stage.tmp"
    staged_path.write_bytes(b"staged\n")
    staged_status = staged_path.stat()
    staged_identity = (staged_status.st_dev, staged_status.st_ino)
    state = _direct_output_state(output_path)
    state.staged_name = staged_path.name
    state.staged_identity = staged_identity
    state.staged_path_identity = staged_identity
    try:
        coding_reliability._cleanup_artifacts(state)

        assert state.staged_name is not None
        assert state.staged_name.startswith(".trackgen-retired-")
        assert state.staged_path_identity == staged_identity
        quarantine = tmp_path / state.staged_name
        assert (quarantine.stat().st_dev, quarantine.stat().st_ino) == (
            staged_identity
        )
    finally:
        os.close(state.directory.fd)


def test_cleanup_aggregates_every_recovery_diagnostic(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "output.csv"
    state = _direct_output_state(output_path)
    state.staged_name = ".output.stage.tmp"
    state.staged_path_identity = (11, 12)
    state.backup_name = ".output.backup.bak"
    state.backup_identity = (21, 22)

    def fail_with_recovery(_state, name, expected_identity):
        quarantine = tmp_path / f".trackgen-retired-{name.strip('.')}"
        raise RuntimeError(
            f"recovery at {quarantine}; "
            f"(dev, ino)=({expected_identity[0]}, {expected_identity[1]})"
        )

    monkeypatch.setattr(
        coding_reliability,
        "_capture_then_classify_entry",
        fail_with_recovery,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            coding_reliability._cleanup_artifacts(state)
    finally:
        os.close(state.directory.fd)

    diagnostic = str(raised.value)
    assert ".trackgen-retired-output.stage.tmp" in diagnostic
    assert "(dev, ino)=(11, 12)" in diagnostic
    assert ".trackgen-retired-output.backup.bak" in diagnostic
    assert "(dev, ino)=(21, 22)" in diagnostic


def test_python310_rollback_detail_remains_in_primary_message():
    class LegacyPrimaryError(RuntimeError):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    primary = LegacyPrimaryError("primary publication failure")
    recovery = RuntimeError(
        "recovery at /tmp/.trackgen-retired-one; (dev, ino)=(31, 32)"
    )

    coding_reliability._note_rollback_error(primary, recovery)

    assert isinstance(primary, LegacyPrimaryError)
    assert "primary publication failure" in str(primary)
    assert "/tmp/.trackgen-retired-one" in str(primary)
    assert "(dev, ino)=(31, 32)" in str(primary)


def test_recovery_path_uses_renamed_anchored_parent(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "original-parent"
    relocated_parent = tmp_path / "relocated-parent"
    parent.mkdir()
    source_path = parent / ".output.stage.tmp"
    source_path.write_bytes(b"expected stage\n")
    expected_status = source_path.stat()
    expected_identity = (expected_status.st_dev, expected_status.st_ino)
    parked_expected = tmp_path / "parked-expected-stage"
    captured_foreign = tmp_path / "captured-foreign"
    captured_foreign.write_bytes(b"captured foreign\n")
    captured_status = captured_foreign.stat()
    captured_identity = (captured_status.st_dev, captured_status.st_ino)
    refill_foreign = tmp_path / "refill-foreign"
    refill_foreign.write_bytes(b"refill foreign\n")
    state = _direct_output_state(source_path)
    real_status = coding_reliability._entry_status
    real_rename = coding_reliability._rename_noreplace_at
    replaced = False
    quarantine_name = None

    def status_then_replace(directory, name):
        nonlocal replaced
        status = real_status(directory, name)
        if not replaced and name == source_path.name:
            source_path.rename(parked_expected)
            captured_foreign.rename(source_path)
            replaced = True
        return status

    def capture_then_relocate(directory, source_name, destination_name):
        nonlocal quarantine_name
        result = real_rename(directory, source_name, destination_name)
        if source_name == source_path.name:
            quarantine_name = destination_name
            parent.rename(relocated_parent)
            anchored_parent = Path(f"/proc/self/fd/{directory.fd}")
            os.link(refill_foreign, anchored_parent / source_name)
        return result

    monkeypatch.setattr(coding_reliability, "_entry_status", status_then_replace)
    monkeypatch.setattr(
        coding_reliability,
        "_rename_noreplace_at",
        capture_then_relocate,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            coding_reliability._capture_then_classify_entry(
                state,
                source_path.name,
                expected_identity,
            )
    finally:
        os.close(state.directory.fd)

    assert replaced
    assert quarantine_name is not None
    quarantine_path = relocated_parent / quarantine_name
    assert quarantine_path.exists()
    assert (quarantine_path.stat().st_dev, quarantine_path.stat().st_ino) == (
        captured_identity
    )
    assert str(quarantine_path) in _exception_text(raised.value)
