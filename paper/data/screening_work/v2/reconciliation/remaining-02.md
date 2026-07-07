# Remaining evidence reconciliation 02

## Scope and method

This note covers `C0114`, `C0115`, `C0116`, `C0119`, `C0123`, and `C0124`.
It answers only whether an accountable author can now inspect enough direct material
evidence to make a *provisional* v2 decision. It neither changes a sealed rating nor
supersedes the dossier or worksheet.

Candidate metadata was checked in `paper/data/screening_inputs/v2/candidates.csv`;
the governing codebook is `paper/data/screening_inputs/v2/protocol.md` (SHA-256
`d3177ec60cfb8f0c229aa2c471dd1b0c4259a1c45a05a286baf8d19d778aeee6`).
The sealed main-result snapshot is
`09f48afb2f33b6d613aafa4340a9597ae031bcd421418c52ed8ccd2994904f44`, bound to
coordinator snapshot `2fded9335462434c82b4b85dc828fdc31d3ba3b4a876062b8399d2ad9f209b19`.
The relevant current entries are in `drafts/batch-03.md`; the worksheet marks all
six `requires_accountable_author_review`.

The deciding protocol rules are: a positive result requires directly inspected
source-native material evidence and a precise locator; title, topic, and abstract
inference cannot establish inclusion. An `abstract_only` source may support a
non-insufficient exclusion only where the accessible abstract directly and
unambiguously establishes that exclusion. This note uses publisher proceedings,
publisher records, and the public AV-FUZZER project artifact only; index records were
used only to discover links and are not deciding evidence.

## C0114 -- Racing tracks improvisation

- **Metadata and sealed ratings.** DOI [10.1109/CIG.2014.6932899](https://doi.org/10.1109/CIG.2014.6932899), IEEE record [6932899](https://ieeexplore.ieee.org/document/6932899/). Both locked ratings, `A-C0114-03` and `A-C0114-06`, are `excluded / exclude-insufficient-detail / abstract_only`. Their source-specific record is preserved in result files with SHA-256 `821fe9bbcc2c963257996ffa871d40ff0d7104c6a68967dc3d80bca803e93f89` and `6cadd8a14884eb7b4f3a824b44f53e883efced5324138d1b05741ea15e81ab58` respectively.
- **Dossier position.** The batch-03 entry correctly says that music-improvisation-based track generation is mentioned, but no inspectable track representation, algorithm, or artifact was recovered.
- **Direct-source check and limitation.** On 2026-07-01, the IEEE document URL and its documented `rest/document/6932899` endpoint returned the publisher's `IEEE Xplore - Unable to Load Page` response in this environment. The DOI identifies the version of record, but no complete report, author manuscript, or official project artifact was available to inspect. The accessible title/partial abstract cannot determine whether its tracks have a transferable course encoding or whether the source makes an inclusion claim.
- **Source-resolvable now: no.** Keep the provisional `excluded / exclude-insufficient-detail` recommendation. The accountable author should obtain the version-of-record PDF through IEEE access or a provenance-established author manuscript, bind its bytes/digest, and inspect the track representation and generation method before reconsidering `include-1`.

## C0115 -- Personalised track design in car racing games

- **Metadata and sealed ratings.** DOI [10.1109/CIG.2016.7860435](https://doi.org/10.1109/CIG.2016.7860435), IEEE record [7860435](https://ieeexplore.ieee.org/document/7860435/), and Imperial College's official handle [10044/1/39560](http://hdl.handle.net/10044/1/39560). `A-C0115-03` and `A-C0115-04` both lock `excluded / exclude-insufficient-detail / abstract_only`; the sealed result files are respectively `821fe9bbcc2c963257996ffa871d40ff0d7104c6a68967dc3d80bca803e93f89` and `e38eaf2af1d2be9f39c9613a9d22b3e241758c4791a59673b6378d51b828342e`.
- **Dossier position.** It reports only the abstract's claim of personalised generated tracks; the methods and any generated-track artifact remain uninspected.
- **Direct-source check and limitation.** IEEE's document and REST routes returned `IEEE Xplore - Unable to Load Page`; the official Imperial handle redirected to Spiral but returned HTTP 429 on 2026-07-01. No author-hosted manuscript, official software, released track data, or version-pinned companion was recovered. The abstract is not sufficient to establish either a source-native generation method or a qualifying representation.
- **Source-resolvable now: no.** Keep `excluded / exclude-insufficient-detail`. Next action: obtain the Spiral deposit or an author-provided manuscript, record its exact version and digest, and inspect the track-generation representation, personalization inputs, and output geometry before applying the first applicable inclusion rule.

## C0116 -- Controllable Procedural Generation of Race Track Surroundings for Iterative Level Design

- **Metadata and sealed ratings.** DOI [10.1109/COG64752.2025.11114175](https://doi.org/10.1109/COG64752.2025.11114175), IEEE record [11114175](https://ieeexplore.ieee.org/document/11114175/). `A-C0116-02` and `A-C0116-05` both lock `excluded / exclude-appearance-dynamics / abstract_only`; their result files are `f79fa34efd64ce0118977988e3d150715caeb1457459c2d4511bcbda4e745506` and `dc66d8a2eb4b6908f7a61e85669bd45ffab43ce600033646ebe10717e129fa27`.
- **Dossier position.** It identifies the deciding publisher-abstract fact: forests, fields, cities, mountains, assets, and vegetation are generated as surroundings, expressly not as racing mechanics or track geometry.
- **Direct-source check and deciding fact.** The source is a publisher version-of-record record, and the sealed ratings preserve the inspected abstract locator. That description directly identifies appearance-only surroundings rather than a course constraint, representation, or geometry operation. Under the protocol's abstract-only exception, it directly activates `exclude-appearance-dynamics`; no positive inference is made from the title or abstract.
- **Current access limitation.** On 2026-07-01, the IEEE document and `rest/document/11114175` routes returned `IEEE Xplore - Unable to Load Page` here. No complete public primary report or author artifact was retrieved, and therefore no claim about an unobserved method is made. There is no archive pin or artifact digest for the paper body.
- **Source-resolvable now: yes, narrowly for provisional exclusion.** Retain `excluded / exclude-appearance-dynamics / abstract_only`. The accountable author should nevertheless retrieve the complete report and confirm that the generated objects never alter track geometry before locking the adjudication; a full report is required for any change toward inclusion.

## C0119 -- Learn-To-Race: A Multimodal Control Environment for Autonomous Racing

- **Metadata and sealed ratings.** Official ICCV/CVF article page [here](https://openaccess.thecvf.com/content/ICCV2021/html/Herman_Learn-To-Race_A_Multimodal_Control_Environment_for_Autonomous_Racing_ICCV_2021_paper.html), official PDF [here](https://openaccess.thecvf.com/content/ICCV2021/papers/Herman_Learn-To-Race_A_Multimodal_Control_Environment_for_Autonomous_Racing_ICCV_2021_paper.pdf), DOI [10.1109/ICCV48922.2021.00965](https://doi.org/10.1109/ICCV48922.2021.00965). `A-C0119-01` is `included / include-1 / full_text`; `A-C0119-02` is `boundary / boundary / full_text`. Both identify the same ICCV 2021 proceedings PDF, SHA-256 `fe6540c38f19aedc95db9f8992cdc750b3a017e2dcb3796db3ed6c641e765fde`; the two sealed result files are `65276e9e0c2c38b0dd5273eae2ec5dd64a336a1998fff294c7269c06348957ae` and `f79fa34efd64ce0118977988e3d150715caeb1457459c2d4511bcbda4e745506`.
- **Dossier position.** The batch-03 entry recommends `included / include-1`, identifying pp. 3-4, Section 3.2, *Track Generation and Custom Track Construction*, and Figure 2 as the deciding source-native custom-track construction/export locator.
- **Direct-source check and deciding fact.** On 2026-07-01, the official CVF PDF served a byte-range response: HTTP 206, `Content-Range: bytes 0-262143/2816760`, `Last-Modified: Sun, 26 Sep 2021 03:46:25 GMT`, ETag `"2afaf8-5ccddd318a640"`, and `Accept-Ranges: bytes`. The sealed complete-PDF digest and precise full-text locator bind the same static proceedings object. Section 3.2 describes construction and export of custom racing-track geometry; this is source-native emission/serialization of explicit course geometry and meets the earlier `include-1` rule. It is not merely a fixed-course benchmark transfer.
- **Access limitation.** A one-shot complete-PDF request timed out after 60 seconds in this environment, so the current file hash was not recomputed. The official endpoint remains directly reachable and supports range retrieval; the preserved digest, ETag, last-modified value, and page/section locator make the exact report independently inspectable.
- **Source-resolvable now: yes.** Provisional outcome: `included / include-1 / full_text`. Next action: the accountable author should download the complete official PDF, verify SHA-256 `fe6540c38f19aedc95db9f8992cdc750b3a017e2dcb3796db3ed6c641e765fde`, inspect pp. 3-4 and Figure 2, then resolve `A1;A2` through the controlled adjudication process.

## C0123 -- Quality Metrics and Oracles for Autonomous Vehicles Testing

- **Metadata and sealed ratings.** DOI [10.1109/ICST49551.2021.00030](https://doi.org/10.1109/ICST49551.2021.00030), IEEE record [9438556](https://ieeexplore.ieee.org/document/9438556/). `A-C0123-03` and `A-C0123-04` both lock `excluded / exclude-insufficient-detail / abstract_only`; their sealed result files are `821fe9bbcc2c963257996ffa871d40ff0d7104c6a68967dc3d80bca803e93f89` and `e38eaf2af1d2be9f39c9613a9d22b3e241758c4791a59673b6378d51b828342e`.
- **Dossier position.** It correctly limits the question to whether the metrics/oracles are source-natively applied across generated or parameterized course variation, as `include-3` requires.
- **Direct-source check and limitation.** The publisher document and REST routes returned `IEEE Xplore - Unable to Load Page` on 2026-07-01. No full paper, author manuscript, official implementation, dataset, or other authoritative companion was found. A title or an abstract-level account of AV quality metrics cannot establish generated-course application, validation across a course distribution, or a named boundary transfer.
- **Source-resolvable now: no.** Keep `excluded / exclude-insufficient-detail`. Next action: retrieve the full version of record or an author-authenticated manuscript and inspect the metric definitions, inputs, evaluated road/course instances, and experiments; only direct evidence of variation across generated/parameterized courses could support `include-3`.

## C0124 -- AV-FUZZER: Finding Safety Violations in Autonomous Driving Systems

- **Metadata and sealed ratings.** DOI [10.1109/ISSRE5003.2020.00012](https://doi.org/10.1109/ISSRE5003.2020.00012), IEEE record [9251068](https://ieeexplore.ieee.org/document/9251068/). `A-C0124-01` and `A-C0124-03` both lock `excluded / exclude-traffic-only / abstract_only`; their sealed result files are `65276e9e0c2c38b0dd5273eae2ec5dd64a336a1998fff294c7269c06348957ae` and `821fe9bbcc2c963257996ffa871d40ff0d7104c6a68967dc3d80bca803e93f89`.
- **Dossier position.** It records the same direct negative distinction from the publisher abstract: the fuzzer perturbs traffic-participant maneuvers and trajectory parameters on an existing road environment, not road geometry.
- **Pinned project artifact.** The public project [cclinus/AV-Fuzzer commit `7ab615c6eab3223af81b6574a3514061a90600ae`](https://github.com/cclinus/AV-Fuzzer/commit/7ab615c6eab3223af81b6574a3514061a90600ae) names the candidate paper in `README.md` line 47 and provides a version-pinned [source archive](https://codeload.github.com/cclinus/AV-Fuzzer/tar.gz/7ab615c6eab3223af81b6574a3514061a90600ae). Retrieved 2026-07-01, that archive has SHA-256 `944c1c4fbebc52b855e820774f97ba7e9ca09b70cbe06cce14416db5fbc793f3`.
- **Deciding locators and fact.** At that commit, `README.md` lines 1-16 describes genetic scenario generation by mutating environmental conditions and NPC behavior; `carla_sim/simulation.py` lines 36-41 loads the fixed CARLA `Town03` world and gets its map; lines 57-74 read fixed ego, NPC, and pedestrian start/end coordinates from `parameters/spawn.yaml`; lines 60-62 set the ego destination. `carla_sim/GA.py` lines 92-98 and 125-129 mutate and generate NPC behavior sequences. The artifact contains no road, route, corridor, or map-geometry generator. These are source-native traffic/scenario operations on a supplied map.
- **Access and provenance limitation.** IEEE's routes returned `IEEE Xplore - Unable to Load Page` here, so the publisher PDF was not inspected. The public repository itself identifies the paper but does not name its authors in the inspected README; the accountable author should verify its project lineage before final locking. That limitation does not change the observed technical fact in the pinned artifact.
- **Source-resolvable now: yes, for provisional exclusion.** Provisional outcome: `excluded / exclude-traffic-only / official_documentation` if the accountable author confirms the repository as the authoritative companion; otherwise retain the existing `abstract_only` access token while preserving the same exclusion. Next action: confirm repository provenance with a listed author or the paper body, then inspect the pinned files above and resolve `A3` without editing the raw ratings.
