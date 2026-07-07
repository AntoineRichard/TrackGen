# Remaining evidence reconciliation 01

Prepared 2026-07-01. This is an evidence-access note only. It does not alter a
sealed rating, dossier, worksheet, candidate record, or CSV.

## Scope and method

This pass reconciles `C0058`, `C0070`, `C0101`, `C0106`, `C0110`, and
`C0113`. For each, I inspected the frozen candidate metadata, the two sealed
v2 ratings, the current dossier entry, and `screening_inputs/v2/protocol.md`.
The protocol requires directly inspected full text, full text plus a companion,
or version-identifiable official documentation for an inclusion or boundary;
an abstract-only record may support only an unambiguous exclusion. A title,
metadata, or abstract inference is not material evidence for a positive claim.

The main sealed-result snapshot is
`09f48afb2f33b6d613aafa4340a9597ae031bcd421418c52ed8ccd2994904f44`.
`C0110` is a calibration record; its two sealed v2 ratings are in the
calibration snapshot. The source checks below used only publisher/DOI records
and the one identified official author page. No newly retrievable complete
report, author manuscript, release-pinned repository, dataset, or technical
documentation was found.

## C0058 - From Generation to Gameplay: Authoring Race Tracks With Repulsive Curves

- **Metadata and sealed record:** `C0058` is Henrich and Koetter (2025), IEEE
  *Transactions on Games*, DOI `10.1109/TG.2025.3561107`; metadata input
  digest `fb8c09b3bb90b3ec3ccde397203c794b62f6e10c660e16350d056f661562f5e2`.
  Both sealed ratings, `A-C0058-03` and `A-C0058-04`, are `excluded /
  exclude-insufficient-detail / abstract_only`; their `evidence_sha256` field
  is `NR`.
- **Current dossier:** `drafts/batch-02.md`, C0058, retains accountable-author
  review because the DOI/abstract describes race-track generation but no
  inspectable report or implementation was available.
- **Official sources and access:** [DOI](https://doi.org/10.1109/tg.2025.3561107)
  redirects to [IEEE Xplore record 10965488](https://ieeexplore.ieee.org/document/10965488/).
  The version identifier is the version-of-record DOI. On 2026-07-01 the
  official IEEE route returned an access challenge (`HTTP 202`), not report
  bytes; no immutable report digest, author manuscript, or official project
  artifact was inspectable.
- **Locator and deciding fact:** the only located official material is the
  bibliographic record for the DOI. It does not expose the report sections,
  track representation, constraints, algorithm, or generated artifact needed
  to apply `include-1`.
- **Source-resolvable now: no.** Do not infer inclusion from the title or the
  abstract-level description. The accountable author should obtain the
  version-of-record PDF or an author-released, version-bound implementation,
  record its SHA-256, and inspect the track representation/generation sections
  before making a provisional decision. Pending that, retain the dossier's
  conservative `exclude-insufficient-detail` recommendation.

## C0070 - Procedural Content Generation for Games: A Survey

- **Metadata and sealed record:** `C0070` is Hendrikx, Meijer, Van Der Velden,
  and Iosup (2013), ACM TOMM, DOI `10.1145/2422956.2422957`; metadata input
  digest `5527600867872391e5db6d6bc55abec45366ce1a31694660556076edc9a8c482`.
  `A-C0070-01` and `A-C0070-03` are both `excluded /
  exclude-insufficient-detail / abstract_only`. The former binds only an
  abstract-record response (`bd6fa0a260764b4e8add9173e0e7c0362e8116d5b4580d90e692f74ac0e1764d`),
  not report bytes; the latter records `evidence_sha256=NR`.
- **Current dossier:** `drafts/batch-02.md`, C0070, correctly notes that
  `include-4` requires an inspected survey, not a broad PCG-survey title or
  an abstract-level taxonomy claim.
- **Official sources and access:** [DOI](https://doi.org/10.1145/2422956.2422957)
  resolves to the [ACM Digital Library record](https://dl.acm.org/doi/10.1145/2422956.2422957).
  The version identifier is the version-of-record DOI. On 2026-07-01 the ACM
  endpoint returned an AWS WAF challenge (`HTTP 202`); no complete survey PDF,
  official author manuscript, or version-pinned companion was inspectable.
- **Locator and deciding fact:** no section, taxonomy table, or course/road
  example in the primary survey is currently inspectable. The sealed abstract
  record cannot establish a direct survey-gap contribution under `include-4`.
- **Source-resolvable now: no.** The accountable author should retrieve the
  survey PDF from ACM or an author-hosted manuscript, bind its version/digest,
  and inspect the taxonomy and relevant course/track treatment. Retain
  `exclude-insufficient-detail` unless that direct evidence is obtained.

## C0101 - Using Genetic Algorithms for Automating Automated Lane-Keeping System Testing

- **Metadata and sealed record:** `C0101` is Klampfl, Klueck, and Wotawa
  (2022), *Journal of Software: Evolution and Process*, DOI
  `10.1002/smr.2520`; metadata input digest
  `808317a3c53f4d518a364481e025fb17b6cfea441401840d815916e4f102d731`.
  `A-C0101-04` and `A-C0101-06` are both `excluded /
  exclude-insufficient-detail / abstract_only`. Their sealed abstract-record
  digest is `be478bbcd9313f74af8800b427b23e0c8b055a7a77947d95f0ef8c2280eaf400`,
  not a digest of an inspected report.
- **Current dossier:** `drafts/batch-02.md`, C0101, identifies an
  abstract-level claim about parametric road networks and genetic variation of
  road geometry, but explicitly leaves inclusion unresolved for lack of the
  report or an authoritative companion.
- **Official sources and access:** [DOI](https://doi.org/10.1002/smr.2520)
  resolves to the [Wiley record](https://onlinelibrary.wiley.com/doi/10.1002/smr.2520).
  The version identifier is the version-of-record DOI. On 2026-07-01 the
  official Wiley endpoint returned `HTTP 403` with a Cloudflare challenge; no
  PDF/XML body, author manuscript, or author-released code artifact was
  inspectable.
- **Locator and deciding fact:** no methods, representation, control-point
  encoding, generator output, or evaluation section is directly available.
  The abstract-level indication is not enough to establish transferable
  adjacent-domain `include-1`.
- **Source-resolvable now: no.** An accountable author should obtain the
  version-of-record or an official author manuscript/code release, record a
  digest/commit, and inspect the road-geometry generation method and outputs.
  Preserve the current `exclude-insufficient-detail` recommendation until then.

## C0106 - A voyage planning framework for energy performance analysis of autonomous inland waterway vessels

- **Metadata and sealed record:** `C0106` is Zhang, Zhang, Thies, Mao, and
  Ringsberg (2025), *Energy*, DOI `10.1016/j.energy.2025.137906`; metadata
  input digest `66c62b2dd2de364b6bc650759647a5723577fd560c3e963ff1aca0ed481c8eb5`.
  `A-C0106-02` and `A-C0106-06` are both `excluded /
  exclude-insufficient-detail / abstract_only`. Their sealed abstract-response
  digests, respectively
  `5beab3b2bd11c4b16ff01215a179c6666959b91dcc4895829b4c5e4215654539` and
  `ff868bd3facac8544357805fe55d45072b41a5d0b760b4010b085433a06cfcb0`,
  do not bind a full report.
- **Current dossier:** `drafts/batch-03.md`, C0106, states that the accessible
  record discusses path following, energy prediction, river hydraulics, and
  speed optimization on supplied waterways; it does not establish emitted or
  selected route geometry.
- **Official sources and access:** [DOI](https://doi.org/10.1016/j.energy.2025.137906)
  and the [ScienceDirect version-of-record page](https://www.sciencedirect.com/science/article/pii/S0360544225035480)
  identify the publication. On 2026-07-01 the official ScienceDirect endpoint
  returned `HTTP 403` with a Cloudflare challenge. No article body, accepted
  manuscript, data release, or official planning artifact was inspectable, so
  no immutable full-text digest is available.
- **Locator and deciding fact:** the publisher record's highlights/abstract,
  noted in the sealed ratings, are incomplete and cannot establish whether the
  framework only plans on supplied waterways or defines a qualifying route
  generator. They therefore cannot support the more specific
  `exclude-out-of-scope` finding either.
- **Source-resolvable now: no.** The accountable author should obtain the
  article or a version-bound official artifact and inspect the framework input,
  route construction, and output representation. Until that occurs, retain the
  dossier's `exclude-insufficient-detail` recommendation.

## C0110 - CRAG - a combinatorial testing-based generator of road geometries for ADS testing

- **Metadata and sealed record:** `C0110` is Arcaini and Cetinkaya (2024),
  *Science of Computer Programming*, DOI `10.1016/j.scico.2024.103171`;
  metadata input digest
  `53e3cd422ce21013474dfe9ab2501a65f0d5a80a292f1b0b65fd72316fe6addd`.
  Calibration ratings `A-C0110-02` and `A-C0110-05` are both `excluded /
  exclude-insufficient-detail / abstract_only`; their metadata-response digests
  are `46e3b73a0c33234fafdc9feb9dc5feb5c779507c365510cd5578447afe6de23b`
  and `3db61a76232421fc6d4c35cd6cb17c655d18b56201642a41cc7c6d8928904c37`.
  Neither is an artifact digest.
- **Current dossier:** `drafts/batch-03.md`, C0110, correctly treats the title
  as a lead only: representation, combinatorial constraints, generated roads,
  and source-native technical contribution were not inspectable.
- **Official sources and access:** [DOI](https://doi.org/10.1016/j.scico.2024.103171)
  resolves to the [ScienceDirect version-of-record page](https://www.sciencedirect.com/science/article/pii/S0167642324000947).
  The identified [author publication page](https://parcaini.github.io/publications.html)
  lists the work but supplies no report PDF, repository, release, or technical
  documentation. On 2026-07-01 the ScienceDirect page returned `HTTP 403` with
  a Cloudflare challenge; no complete report or authoritative CRAG artifact was
  retrievable.
- **Locator and deciding fact:** the official pages establish bibliographic
  identity only. No method section, road encoding, generator API, generated
  coordinates, or release-pinned implementation is available to inspect.
- **Source-resolvable now: no.** The accountable author should locate the
  article, a linked author manuscript, or an official CRAG repository/release;
  bind the retrieved bytes or commit and inspect its road-geometry generator.
  Retain `exclude-insufficient-detail` in the meantime.

## C0113 - Evolving A Diverse Collection of Robot Path Planning Problems

- **Metadata and sealed record:** `C0113` is Ashlock, Manikas, and Ashenayi
  (2006), IEEE CEC, DOI `10.1109/CEC.2006.1688530`; metadata input digest
  `defc1dfb11873c77db5487cb6514861844b0377c8d0af5f5408257d19c5b795d`.
  `A-C0113-04` and `A-C0113-05` are both `excluded /
  exclude-insufficient-detail / abstract_only`. The latter binds the visible
  record response `59a9f86faec0d3067324c4409f6bc77d89118257ba9c64b32c5d840d13f9e8e8`,
  not report bytes; the former records `evidence_sha256=NR`.
- **Current dossier:** `drafts/batch-03.md`, C0113, identifies an
  abstract-level description of evolved grid path-planning problems but finds
  no complete report to verify a transferable route representation, generator,
  or evaluation.
- **Official sources and access:** [DOI](https://doi.org/10.1109/cec.2006.1688530)
  redirects to [IEEE Xplore record 1688530](https://ieeexplore.ieee.org/document/1688530/).
  The version identifier is the version-of-record DOI. On 2026-07-01 the
  official IEEE route returned an access challenge (`HTTP 202`); no paper PDF,
  author manuscript, or official implementation was inspectable and no
  immutable full-text digest is available.
- **Locator and deciding fact:** no complete primary source is available for
  the problem representation, evolutionary operators, diversity criterion, or
  generated-instance outputs. The indexed abstract cannot prove a
  source-native transferable-adjacent course generator.
- **Source-resolvable now: no.** The accountable author should obtain and
  inspect the CEC paper or an author-hosted manuscript, bind its version/digest,
  and verify the path-problem encoding and evolution procedure. Retain the
  conservative `exclude-insufficient-detail` recommendation pending that work.

## Consolidated next action

All six remain **not source-resolvable for a provisional qualifying or
exclusion decision beyond the current conservative insufficient-detail
recommendation**. The accountable author should make a rights-compliant
retrieval of the specified primary report or an official, version-bound
companion; record the immutable version/commit and SHA-256; inspect the named
technical locators; then decide through the controlled adjudication process.
