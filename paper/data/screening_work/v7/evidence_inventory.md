# V7 Calibration Evidence Inventory

The stable-30 calibration has 30/30 manifest coverage: 26 byte-backed local artifacts
and 4 metadata-only artifacts. The manifest is canonical candidate/artifact order and
binds every local artifact to its SHA-256 and filename under
`paper/data/source_archive/v7`.

## Access And Redistribution

| Classification | Count |
| --- | ---: |
| `full_text` | 14 |
| `official_documentation` | 12 |
| `abstract_only` | 4 |
| `public-redistributable` | 1 |
| `local-restricted` | 25 |
| `metadata-only` | 4 |

Public accessibility does not imply redistribution permission. Local retention is
therefore marked `local-restricted` unless a redistribution basis was established; the
commit-pinned Gymnasium MIT source for C0017 is the sole
`public-redistributable` exception.

C0172 is bound to the exact v1.3.0 Routes URL and retained as v1.3.0. The local
HTML identifies v1.3.0 in its `data-version` and current-version selector, while the
publisher's `rel=canonical` incorrectly targets the corresponding v1.4.0 route.

## Metadata-Only Artifacts

The following four stable candidates have no local bytes in the v7 source archive. The
expected filename is reserved for a future, independently retrieved packet artifact;
it is not present in the archive.

| Candidate | Title | Expected filename |
| --- | --- | --- |
| C0009 | Automatic Track Generation for High-End Racing Games Using Evolutionary Computation | `C0009/loiacono-2011-ieee-t-ciaig-3-3.pdf` |
| C0015 | UN Regulation No. 157 - Automated Lane Keeping Systems (ALKS) | `C0015/un-regulation-no-157-2021.pdf` |
| C0110 | CRAG – a combinatorial testing-based generator of road geometries for ADS testing | `C0110/arcaini-2024-scico-103171.pdf` |
| C0134 | RoadSign at the SBFT 2023 Tool Competition Cyber-Physical Systems Track | `C0134/ayerdi-2023-sbft-road-sign.pdf` |

The structured notes in the manifest distinguish v6-recorded metadata from searches
that are not recorded. They establish inventory status only and do not assert an
unsubstantiated exhaustive search.

## Release State

This is a v7 working inventory. No v7 freeze, release, or reviewer launch has occurred.
