# Screening Inclusion Precedence Design

**Status:** Approved design

**Decision:** Use source-native geometry-operation precedence. `include-1` takes
precedence whenever a report performs at least one qualifying operation on explicit
course geometry. `include-2` is selected only when no qualifying `include-1` operation is established.

## Context

The sealed v4 calibration achieved exact status agreement of 24/30 but repeated the
same criterion disagreement on C0147, C0168, and C0172: one reviewer selected
`include-1`, while the other selected `include-2`. The v4 protocol orders the criteria
but does not state explicitly how to classify a source that both performs an operation
on course geometry and defines a reusable representation, interface, benchmark, or
course set. The sealed v4 decision therefore records systematic ambiguity and blocks
main screening.

This revision changes criterion precedence only. It does not broaden the survey scope,
change the definition of a course, change evidence sufficiency, add vocabulary, alter
the stable calibration sample, or reinterpret v4 ratings after the fact.

## Alternatives Considered

1. **Geometry-operation precedence (selected).** `include-1` wins when any qualifying
   source-native geometry operation is established. This is the smallest clarification
   and preserves the existing ordered criteria.
2. **Move serialization to `include-2`.** Reserve `include-1` for generation and
   transformation. This would change the existing meaning of `include-1` and create a
   new boundary between serialization and other geometry operations.
3. **Allow a compound `include-1+2` value.** This would expand the closed vocabulary,
   complicate agreement reporting, and weaken the single-primary-criterion contract.

## Normative Rule

Reviewers evaluate inclusion criteria in order. A source satisfies `include-1` when it
source-natively synthesizes, samples, selects, places, connects, mutates, repairs,
validates, optimizes, or serializes explicit course geometry or a course distribution.
Explicit geometry includes coordinates, centerlines, widths, boundaries, route or road
graphs, waypoint sequences, gate poses, buoy placements, corridors, and equivalent
spatial traversal constraints.

When a source satisfies `include-1`, the reviewer MUST record `include-1` even if the
same source also defines an interface, representation, dataset, benchmark, competition
course set, simulator contract, or interchange artifact that defines an `include-2`-type
contribution. The additional `include-2`-type contribution MUST be recorded in `notes`
but MUST NOT replace the primary criterion.

`include-2` is selected only when no qualifying `include-1` operation is established.
Merely loading, referencing, displaying, or controlling on supplied fixed
geometry is not an `include-1` operation.

The rule applies to both direct robot-racing sources and transferable adjacent domains.
Adjacent-domain mappings remain required in `notes` where the protocol already
requires them.

## Protocol Changes

Create a new protocol version and coordinator snapshot. Modify only the following
normative areas:

- Refine the `include-1` and `include-2` table entries to state the precedence rule.
- Expand the inclusion-precedence clarification with the positive operation list,
  `include-1`-wins rule, and the negative fixed-geometry case.
- Preserve the one-primary-criterion schema and all existing controlled values.
- Record that the revision responds to the v4 calibration decision without importing
  any v4 rating into a reviewer packet.

The protocol title/version must increment. The new coordinator must reuse the exact
stable 30-candidate calibration selection because no bibliographic or discovery
metadata changed.

## Execution And Blinding

The v5 calibration uses six fresh automated reviewer contexts that were not exposed to
v4 ratings or disagreement discussion. Each context receives only its exact rendered
v5 prompt, protocol, role packet, and independently retrieved public sources. The
role-local structural validator remains required before completion.

The v4 coordinator, release, ratings, and revise decision remain immutable historical
artifacts. They are not eligible inputs to v5 ratings or synthesis.

## Validation And Success Criteria

Before release, automated tests and snapshot validators must establish that:

- the v5 protocol and prompt bytes are canonical and frozen;
- all v5 coordinator files and checksums validate;
- the calibration selection is byte-identical to v4;
- six fresh role stages bind exact prompt/configuration/packet hashes;
- every reviewer output passes role-local and authoritative phase validation;
- all 60 ratings seal into one calibration snapshot; and
- the calibration gate reports exact status and criterion disagreement statistics.

Main screening is released only if exact status agreement is at least 0.80, all ratings
and bindings are valid, and no operational definition or rule boundary causes repeated
disagreement under the revised protocol.
