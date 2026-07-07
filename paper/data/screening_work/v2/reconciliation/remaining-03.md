# Evidence reconciliation: C0125, C0132, C0134, C0141, C0144, and C0151

**Scope.** This source-access note was prepared on 2026-07-01. It answers only
whether an accountable author can now inspect enough direct material evidence to
make a provisional v2 eligibility decision. It does not alter sealed ratings, the
current dossiers, the adjudication worksheet, or any CSV.

## Governing records and test

- Candidate metadata is in `paper/data/screening_inputs/v2/candidates.csv` for the
  six candidate IDs below. The frozen coordinator snapshot is
  `2fded9335462434c82b4b85dc828fdc31d3ba3b4a876062b8399d2ad9f209b19`.
- The governing [v2 protocol](../../screening_inputs/v2/protocol.md) has SHA-256
  `d3177ec60cfb8f0c229aa2c471dd1b0c4259a1c45a05a286baf8d19d778aeee6`.
  Its `material evidence` definition and `Access and evidence rules` require direct
  inspection of the report or an authoritative companion artifact with a precise
  locator. `abstract_only` cannot support `included` or `boundary`; incomplete
  evidence requires `excluded / exclude-insufficient-detail`.
- The applicable sealed result-file digests are `screening-01.csv`
  `65276e9e0c2c38b0dd5273eae2ec5dd64a336a1998fff294c7269c06348957ae`,
  `screening-02.csv`
  `f79fa34efd64ce0118977988e3d150715caeb1457459c2d4511bcbda4e745506`,
  `screening-03.csv`
  `821fe9bbcc2c963257996ffa871d40ff0d7104c6a68967dc3d80bca803e93f89`,
  `screening-05.csv`
  `dc66d8a2eb4b6908f7a61e85669bd45ffab43ce600033646ebe10717e129fa27`,
  and `screening-06.csv`
  `6cadd8a14884eb7b4f3a824b44f53e883efced5324138d1b05741ea15e81ab58`.

## C0125 - EvoScenario: Integrating Road Structures into Critical Scenario Generation for Autonomous Driving System Testing

- **Metadata and current dossier:** Tang, Zhang, Zhou, Zhou, Li, and Xue (2023),
  [DOI 10.1109/ISSRE59848.2023.00054](https://doi.org/10.1109/ISSRE59848.2023.00054),
  IEEE document [10301222](https://ieeexplore.ieee.org/document/10301222/).
  The current [batch-03 dossier](../drafts/batch-03.md) records that only the
  abstract's road-segment statement was inspectable.
- **Sealed v2 ratings:** `A-C0125-03` and `A-C0125-06` are both
  `excluded / exclude-insufficient-detail / abstract_only`. Their locators are the
  IEEE/Semantic Scholar abstract and IEEE document record; the raw reasons state
  that representation, operators, constraints, and evaluation were unavailable.
- **Direct retrieval and limitation:** The version-of-record DOI remains the only
  primary report identified. On 2026-07-01 its DOI route resolved to IEEE but did
  not return report bytes; the registered IEEE stamp endpoint
  `https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10301222` returned
  HTTP 418 with an HTML access response, not a PDF. The previously found public
  GitHub project named `EvoScenario` is not linked by the report metadata to any
  of the six paper authors and is not evidence for this candidate.
- **Deciding fact:** No directly inspectable primary report or author/official
  companion establishes how the named sequential road segments are represented,
  generated, constrained, or evaluated.
- **Source-resolvable now: no.** Retain provisional
  `excluded / exclude-insufficient-detail / abstract_only`. An accountable author
  should obtain the IEEE version of record or a lawful author-hosted manuscript,
  bind its bytes to a SHA-256, and inspect the road-representation and generation
  sections before considering `include-1`.

## C0132 - The World Robotic Sailing Championship, a Competition to Stimulate the Development of Autonomous Sailboats

- **Metadata and current dossier:** Le Bars and Jaulin (2015),
  [DOI 10.1109/OCEANS-GENOVA.2015.7271767](https://doi.org/10.1109/OCEANS-GENOVA.2015.7271767),
  IEEE document [7271767](https://ieeexplore.ieee.org/document/7271767/).
  The current [batch-03 dossier](../drafts/batch-03.md) says the accessible record
  does not expose an inspectable course definition or competition course set.
- **Sealed v2 ratings:** `A-C0132-03` and `A-C0132-05` are both
  `excluded / exclude-insufficient-detail / abstract_only`; one sealed locator
  reached only the abstract and visible introductory paragraph, and the other only
  bibliographic metadata.
- **Direct retrieval and limitation:** The official World Robotic Sailing landing
  page, [roboticsailing.org](https://www.roboticsailing.org/), still identifies its
  2015 championship site as `http://www.wrsc2015.com/`. That archive route returned
  HTTP 410 on 2026-07-01, without rules, a buoy layout, a course set, or a report.
  Later rules cannot be substituted for this 2015 report absent a source-native
  link. The IEEE version of record remains closed in the inspected public routes.
- **Deciding fact:** No inspected source tied to this report defines ordered marks,
  buoy coordinates, route tolerances, parameterization, or a citable transfer.
- **Source-resolvable now: no.** Retain provisional
  `excluded / exclude-insufficient-detail / abstract_only`. Next, recover the
  2015 rules or the full paper from the organizers/authors or IEEE, retain the
  retrieved bytes and digest, and inspect the course-definition and scoring sections.

## C0134 - RoadSign at the SBFT 2023 Tool Competition Cyber-Physical Systems Track

- **Metadata and current dossier:** Ayerdi, Arrieta, and Illarramendi (2023),
  [DOI 10.1109/SBFT59156.2023.00006](https://doi.org/10.1109/SBFT59156.2023.00006),
  IEEE document [10190387](https://ieeexplore.ieee.org/document/10190387/).
  The [batch-03 dossier](../drafts/batch-03.md) correctly records that the individual
  report was previously only abstract-accessible.
- **Sealed v2 ratings:** `A-C0134-01` and `A-C0134-03` are both
  `excluded / exclude-insufficient-detail / abstract_only`.
- **Direct primary companion now inspected:** The official SBFT 2023 CPS competition
  report, [SBFT Tool Competition 2023 - Cyber-Physical Systems Track](https://sites.mdu.se/download/18.309552f318f0faf8277bd26/1714383875311/SBFT_Tool_Competition_2023_-_Cyber-Physical_Systems_Track.pdf),
  is a 4-page PDF with DOI
  [10.1109/SBFT59156.2023.00010](https://doi.org/10.1109/SBFT59156.2023.00010).
  Retrieved 2026-07-01, it has PDF SHA-256
  `a2a5a8682b17e3e47733156dfaed8ac6d1e0afaf47299715e85d83d95a87d66c`;
  PDF metadata records creation `2023-07-21` and modification `2023-07-26`.
- **Deciding locators and facts:** PDF p. 2 (proceedings p. 46), Section II.A,
  `Goal`, states that test generators generate challenging virtual roads; the same
  page, Section II.A and Figure 1 discussion, defines a virtual road as a sequence
  of 2-D road points interpolated with cubic splines, with first and last points as
  start and target, and requires non-self-intersection, no overly sharp turns, and
  map containment. On PDF p. 2 (proceedings p. 46), Section II.D, `Tools`, the
  source specifically identifies RoadSign as combining diversity-promoting seeding
  with multi-objective optimization that maximizes road features. Its reference 22
  binds that tool description to Ayerdi, Arrieta, and Illarramendi's RoadSign report.
- **Decision rationale:** This is directly inspected, official companion evidence
  of a source-native road generator's representation, validity constraints, and
  optimization process. The virtual road is an ordered, interpolated spatial route
  whose geometry governs simulated vehicle traversal. The named transferable-domain
  mapping is therefore direct: point sequences plus spline corridor and validity
  constraints map to course coordinates and feasible course geometry. It satisfies
  `include-1`, not merely a topical claim.
- **Source-resolvable now: yes.** The accountable author can make the provisional
  decision `included / include-1 / full_text_and_supplement` after re-inspecting the
  companion locators and confirming the PDF digest. The next controlled step is to
  record that evidence in an append-only adjudication that resolves the two sealed
  abstract-only ratings; do not edit either sealed result.

## C0141 - Interactive Evolution for the Procedural Generation of Tracks in a High-End Racing Game

- **Metadata and current dossier:** Cardamone, Loiacono, and Lanzi (2011),
  [DOI 10.1145/2001576.2001631](https://doi.org/10.1145/2001576.2001631).
  The [batch-03 dossier](../drafts/batch-03.md) records no inspectable full paper or
  authoritative implementation.
- **Sealed v2 ratings:** `A-C0141-02` and `A-C0141-03` are both
  `excluded / exclude-insufficient-detail / abstract_only`; their retrieval notes
  identify the ACM record and Politecnico di Milano handle
  [11311/609170](http://hdl.handle.net/11311/609170).
- **Direct retrieval and limitation:** On 2026-07-01, the institutional handle
  redirected to the official repository but returned an HTTP 403 Cloudflare challenge
  before the item metadata or a bitstream could be inspected. The version-of-record
  ACM route is likewise access-controlled. No author-maintained implementation or
  manuscript was found in the primary/official-source pass.
- **Deciding fact:** The current public evidence never exposes a track encoding,
  evolutionary operators, produced geometry, or user-evaluation material from the
  candidate report.
- **Source-resolvable now: no.** Retain provisional
  `excluded / exclude-insufficient-detail / abstract_only`. An accountable author
  should retrieve the ACM paper or a lawful manuscript, preserve a digest, and inspect
  the track representation and interactive-evolution method before an `include-1`
  determination.

## C0144 - GenRL at the SBST 2022 Tool Competition

- **Metadata and current dossier:** Starace, Romdhana, and Di Martino (2022),
  [DOI 10.1145/3526072.3527533](https://doi.org/10.1145/3526072.3527533),
  ACM landing page [GenRL at the SBST 2022 Tool Competition](https://dl.acm.org/doi/10.1145/3526072.3527533).
  The [batch-03 dossier](../drafts/batch-03.md) records no full report or official
  implementation.
- **Sealed v2 ratings:** `A-C0144-02` and `A-C0144-06` are both
  `excluded / exclude-insufficient-detail / abstract_only`. The sealed records name
  the University of Naples repository item `11588/894581`, but it requires a request.
- **Direct retrieval and limitation:** The authoritative ACM PDF endpoint
  [dl.acm.org/doi/pdf/10.1145/3526072.3527533](https://dl.acm.org/doi/pdf/10.1145/3526072.3527533)
  returned HTTP 403 and a 5,494-byte Cloudflare HTML challenge on 2026-07-01, not
  report bytes. The retrieved challenge response SHA-256 is
  `16272b2f0a5485ca65756a861332a2dccaae527b700552cb2783b6f67340881e`.
  The university item remains a request-only access record; no author/project code
  linked to the named report was inspectable.
- **Deciding fact:** There is no direct technical material for GenRL's road encoding,
  generated geometry, validity testing, or optimization process. Its topical abstract
  cannot carry an inclusion decision.
- **Source-resolvable now: no.** Retain provisional
  `excluded / exclude-insufficient-detail / abstract_only`. Obtain the ACM open
  version or an authorized repository copy, bind it to a SHA-256, and inspect the
  generation and validation sections before adjudication.

## C0151 - Procedural Generation of Road Paths for Driving Simulation

- **Metadata and current dossier:** Campos, Leitão, and Coelho (2015),
  [DOI 10.4018/IJCICG.2015070103](https://doi.org/10.4018/IJCICG.2015070103),
  publisher landing page [IGI Global article 147171](https://www.igi-global.com/viewtitle.aspx?TitleId=147171).
  The [batch-04 dossier](../drafts/batch-04.md) records only abstract/excerpt material.
- **Sealed v2 ratings:** `A-C0151-01` and `A-C0151-05` are both
  `excluded / exclude-insufficient-detail / abstract_only`.
- **Direct retrieval and limitation:** The official INESC TEC record is now reachable
  at [handle 123456789/5179](http://repositorio.inesctec.pt/handle/123456789/5179).
  Its DSpace REST record identifies the candidate report and exposes bitstream
  `0cb95ea8-ba88-4e8d-b789-6d58596658e0`, named `P-00M-WVM.pdf`, 558,912 bytes:
  [bitstream content endpoint](https://repositorio.inesctec.pt/server/api/core/bitstreams/0cb95ea8-ba88-4e8d-b789-6d58596658e0/content).
  On 2026-07-01 that endpoint returned HTTP 401 JSON (`Authentication is required`),
  so no report bytes could be inspected or hashed. The same repository API records
  a `canRequestACopy` authorization but no public `canDownload` authorization.
- **Deciding fact:** The newly visible institutional metadata confirms a likely
  manuscript exists, but it supplies no directly inspectable method, road-path
  representation, constraints, outputs, or evaluation. Metadata and abstracts are
  not material evidence under the protocol.
- **Source-resolvable now: no.** Retain provisional
  `excluded / exclude-insufficient-detail / abstract_only`. The accountable author
  should use the repository's request-a-copy route or lawful publisher access, retain
  the retrieved PDF and SHA-256, then inspect the method and results for an emitted
  road-path representation before considering `include-1`.
