# High-Difficult Evidence Acquisition

Retrieved on 2026-07-06 from the CSV-aware selection of all 36 `priority=high` rows in `main_evidence_remaining_queue.csv`. Results are limited to sources that could be downloaded and verified in this environment; they do not establish exhaustive public availability.

Counts: 36 selected, 14 full-text successes, 22 unresolved access limitations, 0 duplicate IDs.

## Verified Full Text

| ID | Title | Verified source |
| --- | --- | --- |
| C0024 | Diversity-guided Search Exploration for Self-driving Cars Test Generation through Frenet Space Encoding | arXiv: https://arxiv.org/abs/2401.14682 |
| C0052 | Procedural modeling of cities | UC Berkeley EECS: https://people.eecs.berkeley.edu/~sequin/CS285/PAPERS/Parish_Muller01.pdf |
| C0054 | Interactive procedural street modeling | Peter Wonka author page: https://peterwonka.net/Publications/pdfs/2008.SG.Chen.InteractiveProceduralStreetModeling.pdf |
| C0069 | Experience-Driven Procedural Content Generation | Georgios Yannakakis author page: https://yannakakis.net/wp-content/uploads/2015/11/PID3821875.pdf |
| C0102 | Testing Autonomous Robot Control Software Using Procedural Content Generation | White Rose: https://eprints.whiterose.ac.uk/id/eprint/103118/ |
| C0107 | A Search-Based Framework for Automatic Generation of Testing Environments for Cyber-Physical Systems | arXiv: https://arxiv.org/abs/2203.12138 |
| C0121 | Environment as Policy: Learning to Race in Unseen Tracks | arXiv v3: https://arxiv.org/abs/2410.22308v3 |
| C0126 | Lanelet2: A high-definition map framework for the future of automated driving | KIT: https://www.mrt.kit.edu/z/publ/download/2018/Poggenhans2018Lanelet2.pdf |
| C0130 | MAVRL: Learn to Fly in Cluttered Environments With Varying Speed | arXiv: https://arxiv.org/abs/2402.08381 |
| C0132 | The World Robotic Sailing Championship, a Competition to Stimulate the Development of Autonomous Sailboats | ENSTA author page: https://webperso.ensta.fr/lebars/paper_wrsc_2013_oceans_2015.pdf |
| C0137 | Preliminary Evaluation of Path-aware Crossover Operators for Search-Based Test Data Generation for Autonomous Driving | KAIST COINSE: https://coinse.github.io/publications/pdfs/Han2021vp.pdf |
| C0143 | Wasserstein generative adversarial networks for online test generation for cyber physical systems | arXiv: https://arxiv.org/abs/2205.11060 |
| C0146 | Search-based Generation of Waypoints for Triggering Self-Adaptations in Maritime Autonomous Vessels | Simula author page: https://web-backend.simula.no/sites/default/files/2025-08/GECCO2025_Paper_Waypoints_search%20.pdf |
| C0152 | Procedural Generation of High-Definition Road Networks for Autonomous Vehicle Testing and Traffic Simulations | SAE: https://saemobilus.sae.org/articles/procedural-generation-high-definition-road-networks-autonomous-vehicle-testing-traffic-simulations-12-06-01-0007 |

All success artifacts were checked as nonempty PDFs with `file`, `pdfinfo`, and first-page `pdftotext`; the manifest records their SHA-256 values. C0121 is pinned to the archived arXiv v3 author manuscript. C0146 and C0152 are marked public-redistributable only because their PDFs explicitly state CC BY 4.0. All other archived copies are local-restricted. The failure ledger's `attempted_urls` field is a JSON array of dated route records with exact URLs and observed statuses; its limitations remain non-exhaustive.

## Unresolved Core Generation

1. C0003: TrackGen: An interactive track generator for TORCS and Speed-Dreams
2. C0012: Comparative Analysis of Metaheuristic Algorithms for Procedural Race Track Generation in Games
3. C0022: Automatically Testing Self-Driving Cars with Search-Based Procedural Content Generation
4. C0044: Optimization-based Path Planning for an Autonomous Vehicle in a Racing Track
5. C0058: From Generation to Gameplay: Authoring Race Tracks With Repulsive Curves
6. C0109: Frenetic-lib: An extensible framework for search-based generation of road structures for ADS testing
7. C0110: CRAG – a combinatorial testing-based generator of road geometries for ADS testing
8. C0114: Racing tracks improvisation
9. C0115: Personalised track design in car racing games
10. C0116: Controllable Procedural Generation of Race Track Surroundings for Iterative Level Design
11. C0125: EvoScenario: Integrating Road Structures into Critical Scenario Generation for Autonomous Driving System Testing
12. C0127: Scenario Factory: Creating Safety-Critical Traffic Scenarios for Automated Vehicles
13. C0133: Analysis of Road Representations in Search-Based Testing of Autonomous Driving Systems
14. C0135: Spirale at the SBFT 2023 Tool Competiton - Cyber-Physical Systems Track
15. C0141: Interactive Evolution for the Procedural Generation of Tracks in a High-End Racing Game
16. C0151: Procedural Generation of Road Paths for Driving Simulation

## Unresolved Supporting Context

1. C0036: Generating a Racing Line for an Autonomous Racecar Using Professional Driving Techniques
2. C0039: Optimization of the Driving Line on a Race Track
3. C0041: Time-optimal trajectory planning for a race car considering variable tyre-road friction coefficients
4. C0043: Learning at the Racetrack: Data-Driven Methods to Improve Racing Performance Over Multiple Laps
5. C0068: Search-Based Procedural Content Generation: A Taxonomy and Survey
6. C0070: Procedural Content Generation for Games: A Survey

C0110 remains unresolved specifically for the cited Science of Computer Programming article. The unrelated local 2024 SBFT tool paper was not used as a substitute.
