# Survey Structure Analysis

**Analysis date:** 2026-06-29  
**Purpose:** identify transferable structural practices from six IJRR and Science Robotics exemplars before drafting the course-generation survey.

## Evidence and citation-count policy

Article metadata was checked against the current journal landing page and the Crossref work record. Full structure was inspected from the version of record when accessible or from an author/institutional manuscript whose title and DOI match the journal record. Crossref's `is-referenced-by-count` was queried on 2026-06-29. The count is a mutable snapshot of references deposited with Crossref, not a complete citation total.

Citation counts are recorded only to explain why these established, visible reviews were selected as structural exemplars. A high count is not evidence that a taxonomy, table design, review method, or research agenda is scientifically correct. Structural choices below are adopted only when they fit this survey's scope and evidence needs.

| Exemplar | Journal metadata checked | Crossref references-to count | Crossref indexed timestamp |
| --- | --- | ---: | --- |
| [Reinforcement learning in robotics: A survey](https://doi.org/10.1177/0278364913495721) | IJRR 32(11), 1238-1274; online 2013-08-23 | [2354](https://api.crossref.org/works/10.1177%2F0278364913495721) | 2026-06-26T15:10:40Z |
| [Human motion trajectory prediction: a survey](https://doi.org/10.1177/0278364920917446) | IJRR 39(8), 895-935; online 2020-06-07 | [626](https://api.crossref.org/works/10.1177%2F0278364920917446) | 2026-06-24T13:14:00Z |
| [Dynamic movement primitives in robotics: A tutorial survey](https://doi.org/10.1177/02783649231201196) | IJRR 42(13), 1133-1184; online 2023-09-23 | [201](https://api.crossref.org/works/10.1177%2F02783649231201196) | 2026-06-16T17:01:57Z |
| [Social robots for education: A review](https://doi.org/10.1126/scirobotics.aat5954) | Science Robotics 3(21), eaat5954; publisher copy dated 2018-08-15, Crossref published date 2018-08-22 | [1250](https://api.crossref.org/works/10.1126%2Fscirobotics.aat5954) | 2026-06-29T14:28:50Z |
| [Biohybrid actuators for robotics: A review of devices actuated by living cells](https://doi.org/10.1126/scirobotics.aaq0495) | Science Robotics 2(12), eaaq0495; institutional record dated 2017-11-29, Crossref published date 2017-11-22 | [503](https://api.crossref.org/works/10.1126%2Fscirobotics.aaq0495) | 2026-06-27T09:48:44Z |
| [A review of collective robotic construction](https://doi.org/10.1126/scirobotics.aau8479) | Science Robotics 4(28), eaau8479; published 2019-03-13 | [196](https://api.crossref.org/works/10.1126%2Fscirobotics.aau8479) | 2026-06-25T06:32:25Z |

The two date discrepancies are retained rather than silently reconciled: the publisher-layout Social Robots copy says 15 August while Crossref reports 22 August, and the Max Planck Biohybrid record says 29 November while Crossref reports 22 November.

## 1. Reinforcement learning in robotics: A survey

**Metadata and inspected text.** Jens Kober, J. Andrew Bagnell, and Jan Peters. The [SAGE journal record](https://journals.sagepub.com/doi/10.1177/0278364913495721) was checked against Crossref; structure was inspected in the [CMU Robotics Institute author copy](https://www.ri.cmu.edu/pub_files/2013/7/Kober_IJRR_2013.pdf).

**Section sequence.**

1. Introduction, positioning RL relative to machine learning, optimal control, and robotics.
2. Concise introduction to RL, including goals, average reward, value functions, policy search, their comparison, and function approximation.
3. Challenges in robot RL: dimensionality, real-world samples, model uncertainty, and goal specification.
4. Tractability through representation.
5. Tractability through prior knowledge.
6. Tractability through models.
7. Ball-in-a-cup case study from task/reward and policy representation through simulation and real-robot results.
8. Discussion: open research questions, practical challenges, and lessons from robotics for RL.

**Scope and relation to prior reviews.** The scope is behavior generation in robotics, with emphasis on results obtained on physical robots and tasks beyond standard RL benchmarks. The article explicitly addresses both robotics and RL audiences. It distinguishes itself from general RL treatments through the constraints of continuous, high-dimensional, partially observed, expensive, and hard-to-repeat robot interaction. It does not provide a systematic search protocol or a dedicated prior-survey comparison table.

**Taxonomy depth.** This is an argument-led organization rather than a formal tree. It first contrasts model-based/model-free and value-function/policy-search choices, then organizes successful robotics work by three tractability mechanisms: representation, prior knowledge, and models. Subsections add another level for discretization, approximation, structured policies, demonstration, task structure, exploration, simulation bias, and forward-model methods.

**Comparison-table axes.** Tables map an approach family to the publications that employ it. Separate tables cover value-function methods, policy-search methods, representations, forms of prior knowledge, and model use. This gives excellent lineage coverage but weak direct comparison of evaluation conditions, code, data, or reproducibility.

**Evaluation and reproducibility.** The ball-in-a-cup case study makes design decisions concrete and reports simulation and real-robot stages. The discussion directly identifies costly experiments, hardware differences, lack of reproducibility, and inconsistent evaluation as field limitations. It calls for shared real and simulated setups and public skill datasets, but the review itself does not expose a reproducible literature-search protocol.

**Research agenda.** The agenda follows from the earlier challenge decomposition: automatic representations, learned rewards, appropriate prior knowledge, tighter perception integration, lower parameter sensitivity, robust use of imperfect models, better dataset reuse, and comparable experiments. Each question is tied to a limitation established in the body.

## 2. Human motion trajectory prediction: a survey

**Metadata and inspected text.** Andrey Rudenko, Luigi Palmieri, Michael Herman, Kris M. Kitani, Dariu M. Gavrila, and Kai O. Arras. The [SAGE journal record](https://journals.sagepub.com/doi/10.1177/0278364920917446) was checked against Crossref; structure was inspected in the [author manuscript on arXiv](https://arxiv.org/abs/1905.06113).

**Section sequence.**

1. Introduction: terminology, application domains, and related surveys.
2. Taxonomy.
3. Physics-based approaches.
4. Pattern-based approaches.
5. Planning-based approaches.
6. Contextual cues.
7. Motion-prediction evaluation: metrics and datasets.
8. Discussion: benchmarking, modeling approaches, application domains, and future directions.
9. Conclusions.

**Scope and relation to prior reviews.** The paper explicitly limits itself to ground-level 2D trajectory prediction for pedestrians while including cyclists and vehicles. Video-frame prediction, articulated motion, actions, and activities are out of scope. The related-surveys subsection compares robotics, intelligent-vehicle, and computer-vision reviews, then argues that earlier classifications mix modeling mechanisms and contextual awareness. The new contribution is to keep those dimensions orthogonal.

**Taxonomy depth.** The first axis has three method families: physics-based, pattern-based, and planning-based. Each divides again, for example single/multiple models, sequential/non-sequential patterns, and forward/inverse planning. The second axis codes target-agent cues, dynamic-environment awareness, and static-environment awareness, each with multiple levels. Explicit classification rules handle ambiguous methods.

**Comparison-table axes.** The metrics table separates geometric, probabilistic, sampling-based, and robustness measures and explains their behavior. Dataset tables compare location, agent type, sensors, scene description, duration/track volume, annotations, and sampling rate. Figures cross modeling family, context, and publication trends.

**Evaluation and reproducibility.** Evaluation is a first-class section rather than a paragraph inside method review. The authors analyze inconsistent metric definitions, arbitrary horizons, dataset limitations, sensing assumptions, robustness, runtime, and missing context annotations. They recommend reporting geometric and probabilistic metrics across horizons and scene complexity, realistic sensing perturbations, runtime/complexity, and standard benchmarks such as TrajNet.

**Research agenda.** Three questions introduced up front structure the discussion: whether evaluation follows best practice, whether modeling families have converged, and whether prediction is solved. The answers synthesize evidence from metrics, datasets, method properties, and application requirements before proposing better context use, goal inference, generalization, robustness, and prediction-planning integration.

## 3. Dynamic movement primitives in robotics: A tutorial survey

**Metadata and inspected text.** Matteo Saveriano, Fares J. Abu-Dakka, Aljaz Kramberger, and Luka Peternel. The [SAGE journal record](https://journals.sagepub.com/doi/10.1177/02783649231201196) was checked against Crossref; structure was inspected in the [author manuscript on arXiv](https://arxiv.org/abs/2102.03861).

**Section sequence.**

1. Introduction: existing surveys/tutorials, systematic review process, taxonomy, and contributions.
2. DMP formulations: discrete, orientation, periodic, and geometry-aware formulations.
3. Extensions: generalization, joining, online adaptation, and related formulations.
4. Integration in larger frameworks: manipulation, variable impedance, RL, deep learning, and lifelong learning.
5. Application scenarios: interaction, co-manipulation, assistance/rehabilitation, teleoperation, high-degree-of-freedom systems, recognition, driving, and field robotics.
6. Discussion: selection guidelines, resources/code, limitations, and open issues.
7. Concluding remarks.

**Scope and relation to prior reviews.** The article combines a mathematical tutorial with a survey of DMP integration and applications. Table 1 compares previous tutorials/reviews by covered topics and description, then argues that they focus on a research group, formulation, or application subset. This paper claims broader coverage and a unified notation.

**Taxonomy depth.** A top-level split separates tutorial and survey. The tutorial branches into formulations and extensions; the survey branches into integration frameworks and applications. Each branch has two or more levels. The taxonomy doubles as the paper outline, which improves navigation but is tailored to a mature method family.

**Comparison-table axes.** Table 1 uses prior survey/tutorial, topics, and description. A formulation table summarizes mathematical variants. The released-code table uses approach, author, language, and description. The limitations table uses limitation, related work, and solved/partially-solved status.

**Evaluation and reproducibility.** The authors searched Scopus for "Dynamic Movement Primitive" on 2020-11-25 and refined the search on 2023-06-20. They report 1223 initial papers and 321 analyzed DMP papers among 373 references. Selection included manual judgments of technical quality and significance. They inventory third-party implementations and release code at [dmp-codes-collection](https://gitlab.com/dmp-codes-collection). The explicit dates and counts are useful; the reliance on venue prestige or citation count as a tie-breaker is not appropriate for this survey's scientific inclusion decisions.

**Research agenda.** Guidelines first explain which formulation fits which application. Open issues then cover implicit time dependence, missing stochastic information, closed-loop stability/passivity, high-dimensional inputs, and multi-attractor behavior. A status table separates addressed from partially addressed limitations.

## 4. Social robots for education: A review

**Metadata and inspected text.** Tony Belpaeme, James Kennedy, Aditi Ramachandran, Brian Scassellati, and Fumihide Tanaka. Metadata was checked against the [Science Robotics record](https://www.science.org/doi/10.1126/scirobotics.aat5954), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33141719/), Crossref, and the [UGent author repository record](https://biblio.ugent.be/publication/8571588). The repository attachment was access-restricted, so section structure was inspected from a [public publisher-layout copy](https://pybeebee.github.io/robotrights/readings/M9_Readings.pdf) and crosschecked against the official metadata.

**Section sequence.**

1. Introduction.
2. Benefits of social robots as tutoring agents.
3. Technical challenges of building robot tutors.
4. Review method and meta-analysis.
5. Efficacy of robots in education, including appearance and behavior.
6. Robot roles: tutor/teacher, peer, and novice.
7. Discussion.

**Scope and relation to prior reviews.** The review includes robots intended to deliver learning through social interaction. It excludes robots used merely as tools for STEM instruction. It distinguishes itself from reviews of virtual pedagogical agents, intelligent tutoring systems, long-term HRI, and narrower education-robot studies. Three questions bound the synthesis: efficacy, embodiment, and interaction role.

**Taxonomy depth.** The classification is shallower than the IJRR exemplars. It organizes evidence by outcome type (cognitive/affective), embodiment and robot properties, and interaction role. Roles divide into tutor/teacher, peer, and novice. This is enough for a focused review but would be too coarse for a cross-domain course-generation survey.

**Comparison-table axes.** Table 1 contrasts common cognitive and affective outcome measures. Figures compare outcome type, robot role, learners per robot, participant demographics, robot platforms, countries, and effect-size distributions. These axes connect design choices to evidence rather than listing papers only.

**Evaluation and reproducibility.** The review states search terms, databases, manually searched HRI venues, a May 2017 cutoff, and inclusion criteria. It reports 101 included papers and 309 study results. Only 81 results contained enough data to compute an effect size, which supports a concrete reporting-quality critique. Extracted variables include study design, conditions, participants, demographics, robot, country, role, learning topic, and effect-size inputs.

**Research agenda.** Recommendations follow from both technical bottlenecks and the meta-analysis: robust child speech/social perception, integrated action selection and behavior generation, longitudinal deployment, one-to-many teaching, curriculum integration, and stronger statistical reporting.

## 5. Biohybrid actuators for robotics: A review of devices actuated by living cells

**Metadata and accessible evidence.** Leonardo Ricotti, Barry Trimmer, Adam W. Feinberg, Ritu Raman, Kevin K. Parker, Rashid Bashir, Metin Sitti, Sylvain Martel, Paolo Dario, and Arianna Menciassi. Metadata and the one-sentence abstract were checked through the [Science Robotics DOI record](https://www.science.org/doi/10.1126/scirobotics.aaq0495), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33157905/), Crossref, and the [Max Planck institutional record](https://pure.mpg.de/view/item_2577815). The verifiable abstract bounds the topic to biohybrid systems that use living cells or tissues to actuate artificial devices.

**Exact retrieval limitation on 2026-06-29.**

- Science.org PDF and ePDF requests returned a Cloudflare challenge rather than article content.
- Crossref's publisher-syndication URL returned HTTP 403; XML content negotiation returned no body.
- PubMed supplies metadata/abstract only. A PubMed Central DOI query and exact-title query returned zero records.
- [Europe PMC's record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1126/scirobotics.aaq0495&format=json) reports `isOpenAccess=N`, `inEPMC=N`, `hasPDF=N`, and no PMCID or full-text URL.
- The Max Planck author-institution record exposes metadata but no file/component in its public item response.
- Targeted title searches across the Sant'Anna, CMU, MIT, and Tufts author-institution domains did not locate an accessible manuscript. A Polytechnique Montreal record linked only to the publisher.
- An OpenAlex-reported repository record was stale/misattributed and did not resolve to this article, so it was not used as evidence.

**Structure fields not claimed.** The full section sequence, placement of scope boundaries, treatment of prior reviews, taxonomy depth, comparison-table axes, evaluation/reproducibility treatment, and research-agenda argument could not be verified from accessible official or author materials. They are therefore not reconstructed from memory or inferred from the title. This exemplar contributes only its verified metadata and broad abstract-level scope to selection context; it does not support any adopted structural decision below.

## 6. A review of collective robotic construction

**Metadata and inspected text.** Kirstin H. Petersen, Nils Napp, Robert Stuart-Smith, Daniela Rus, and Mirko Kovac. The [Science Robotics record](https://www.science.org/doi/10.1126/scirobotics.aau8479) was checked against Crossref; structure was inspected in the [UCL institutional manuscript](https://discovery.ucl.ac.uk/10117871/1/A_Review_of_Collective_Robotic_Construction___Single_File.pdf).

**Section sequence.**

1. Introduction.
2. Construction in nature.
3. Structures and design.
4. Coordination, including centralization/concurrency and representations of agents, actions, states, and goals.
5. Mechanisms and material, including construction materials and robotic platforms.
6. Performance metrics, including construction efficiency/system robustness and complexity/emergence.
7. Conclusion and opportunities.

**Scope and relation to prior reviews.** CRC is explicitly defined as embodied, autonomous, multi-robot systems that modify a shared environment according to high-level user goals. The introduction locates CRC at the intersection of construction, distributed computing, self-organization, and bio-inspired robotics while asserting its coupled design questions. It bounds the field by definition and an overlap figure rather than by comparing prior review articles.

**Taxonomy depth.** The synthesis is deliberately multi-dimensional: biological principles, structure typology (2D, 2.5D, and 3D), additive/removal operations, centralization and concurrency, discrete/continuous/probabilistic state representations, blueprint/functional goals, discrete/continuous materials, binding mechanisms, and platform mobility. These dimensions are not collapsed into one hierarchy.

**Comparison-table axes.** Table 1 maps focus area to biological principle and references. Table 2 maps discrete/continuous material or binder class to demonstrated examples and references. Figures cross colony/robot scale, centralization, platform, material, representation, and target structure.

**Evaluation and reproducibility.** A dedicated metrics section proposes normalized construction output (structure volume per robot volume per time), longevity/depositions, reliability, tolerance and adaptation, and measures of complexity/emergence. It computes example values for aerial and climbing systems. The review does not report a systematic search protocol or code/data release, so reproducibility is analytical rather than corpus-procedural.

**Research agenda.** Opportunities are argued from observed integration limits: robust autonomous emergence, local perception of global structure/stability, reliable mechanisms for 3D manipulation and mobility, hardware/software/material co-design, arbitrary materials, task allocation, and the role of human oversight.

## Cross-exemplar synthesis

Reliable practices shared by the inspectable exemplars are:

1. Bound the object of study before presenting categories.
2. State how adjacent reviews leave a specific gap.
3. Use an explicit taxonomy that also controls paper navigation.
4. Keep method classification distinct from context, representation, or application dimensions.
5. Give metrics, datasets/benchmarks, and reproducibility their own analysis.
6. Build the agenda from diagnosed evidence gaps rather than a generic future-work list.
7. Expose search dates, inclusion rules, and corpus limitations when the review claims systematic coverage.
8. Use comparison axes that support decisions, not only paper-to-category lookup.

The principal adaptation needed here is stronger separation. Course representations, generation mechanisms, metrics, benchmarks, simulator/export constraints, and open problems answer different questions and must not be merged into one "taxonomy" section.

## Adopted survey structure

1. **Introduction and contribution.** Establish why course generation affects generalization, safety, and comparability.
2. **Scope, definitions, and review method.** Define course objects and separate generation from racing-line optimization, planning, control, and perception; report search and screening practice.
3. **Gap relative to prior reviews.** Compare adjacent surveys using explicit coverage axes.
4. **Course representations.** Treat segment/tile, curve, centerline-plus-width, gate-pose, waypoint-graph, world-asset, and simulator-native forms.
5. **Generation mechanisms.** Treat constructive, stochastic, search/evolutionary, learned, environment-design, human-designed, and repair/projection families.
6. **Domain constraints.** Compare ground, aerial, maritime, and adjacent agile robots without using vehicle domain as the primary method taxonomy.
7. **Metrics and reporting protocol.** Separate feasibility, geometry, difficulty, diversity, dynamics, throughput, and reproducibility.
8. **Benchmarks, simulators, and portability.** Separate training distributions from fixed evaluation suites; cover formats, loading, coordinate frames, assets, reset semantics, and validation after import.
9. **Evidence synthesis and reporting gaps.** Compare what current papers actually disclose and release.
10. **Open problems and research agenda.** Derive priorities from missing evidence, transfer failures, and benchmark gaps.

This ordering keeps representations, generation mechanisms, metrics, benchmarks, and open problems as separate top-level sections.

## Adopt / adapt / reject matrix

| Exemplar practice | Decision | Use in this survey | Scientific guardrail |
| --- | --- | --- | --- |
| Early operational scope and exclusions (Rudenko; Belpaeme; Petersen) | Adopt | Define generated course objects and downstream boundary cases in Section 2. | Keep boundary sources when they supply metrics or requirements. |
| Explicit comparison with prior reviews (Rudenko; Saveriano) | Adopt | Use a gap table with representation, generation, validity, metrics, benchmarks, export, and reproducibility axes. | Coverage must be read from the cited review, not inferred from its title. |
| Taxonomy doubles as paper outline (Rudenko; Saveriano) | Adapt | Let stable dimensions guide Sections 4-6. | Do not force orthogonal dimensions into one tree. |
| Orthogonal method and context axes (Rudenko) | Adopt | Keep representation, mechanism, and domain as independent coded dimensions. | Multi-role papers may receive multiple evidence codes with notes. |
| Challenge-to-solution narrative (Kober) | Adapt | Use diagnosed validity and reproducibility problems to motivate method comparisons. | Do not let TrackGen's implementation challenges define the field. |
| Approach-to-publication tables (Kober) | Adapt | Add validity, output, evaluation, code, and portability axes. | A category-only table is insufficient evidence of comparability. |
| Dedicated metrics and dataset analysis (Rudenko) | Adopt | Give metrics and benchmarks separate sections and tables. | Preserve source metric names and distinguish geometric from dynamics-aware measures. |
| Quantitative meta-analysis across heterogeneous studies (Belpaeme) | Adapt | Quantify reporting completeness and corpus composition where denominators are defensible. | Do not pool incompatible outcomes into an effect size. |
| Normalized domain-specific performance measures (Petersen) | Adopt | Seek throughput normalized by batch/hardware plus feasibility, diversity, and simulator-load measures. | State assumptions and avoid comparing incompatible hardware without context. |
| Systematic search dates, counts, and criteria (Saveriano; Belpaeme) | Adopt | Maintain frozen queries, search logs, seed coverage, screening status, and metadata evidence. | Corpus changes remain auditable; unverified bootstrap rows are not evidence. |
| Code/resource inventory (Saveriano) | Adopt | Record official code, assets, formats, and reproducibility fields after verification. | Silence is not evidence that code is unavailable. |
| Citation count or venue prestige as a selection tie-breaker (Saveriano) | Reject | Use relevance, evidence quality, and explicit screening rules instead. | Counts explain exemplar selection only and never validate a scientific choice. |
| Long tutorial derivations before field synthesis (Saveriano) | Reject for the main survey | Put only notation needed to compare representations and metrics in the main text. | Detailed derivations belong in appendices or referenced tutorials. |
| Unverified reconstruction of inaccessible article structure | Reject | Record retrieval limits, as done for Biohybrid, and leave unsupported fields unclaimed. | No structural conclusion may depend on inaccessible evidence. |
| Agenda derived from explicit limitations (all inspectable exemplars) | Adopt | Tie each open problem to coded evidence, benchmark absence, or reproducibility failure. | Separate observed gaps from author inference and proposed research directions. |

