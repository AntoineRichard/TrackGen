Configuration reference
=======================

A ``TrackGenerator`` takes a ``TrackGenConfig`` and a ``GateGenerator`` a
``GateGenConfig``. The core fields (batch, device, relaxation, inflation, spacing, Z
profile, gate counts) apply whatever the generator. A generator's own knobs are its
*Params*: set them through ``generator_params``, or, for the built-in generators, through
the flat fields that carry the same names. This page lists:

- the two config classes and every field (:ref:`config-classes`);
- how ``relax_solver`` and ``z_profile`` select registered pipeline stages
  (:ref:`flat-stage-shortcuts`);
- how ``generator_params`` resolves (:ref:`generator-params-config`);
- each built-in generator's Params, with defaults and rules
  (:ref:`builtin-generator-params`);
- when the generator-specific rules are checked (:ref:`generator-params-validation`);
- the deprecated fields (:ref:`deprecated-config-fields`).

.. _config-classes:

Config classes
--------------

.. autoclass:: track_gen.TrackGenConfig
   :no-members:

.. autoclass:: track_gen.GateGenConfig
   :no-members:

.. _flat-stage-shortcuts:

Pipeline stages: ``relax_solver`` and ``z_profile``
---------------------------------------------------

``TrackGenerator(config)`` builds the default pipeline of
:doc:`/how-it-works/pipeline-preview`, and two flat fields pick its replaceable stages by
name through the plugin registry (``track_gen.plugin.registry``), the way
``generator`` picks the generator:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Field
     - Selects
     - Params of the stage
   * - ``relax_solver``
     - The ``relax`` stage registered under that name: ``"xpbd"`` (default) or a
       plugin's, such as ``"acme.taubin"``. ``relax_enable=False`` selects
       ``relax.none`` instead (``relax_solver`` is still resolved first).
     - ``relax.xpbd``: ``half_width`` and the ``relax_*`` fields, as before. A plugin:
       its ``Params.from_legacy(config)``.
   * - ``z_profile``
     - The ``elevation`` stage registered under that name: ``"flat"`` (default),
       ``"uniform"``, ``"random_walk"``, ``"noise"`` or a plugin's, such as
       ``"acme.sinusoid_hills"``. Any name other than ``"flat"`` lifts with
       ``lift.elevated`` (3D tangents and arc length).
     - The built-ins: the ``z_*`` fields their profile uses, as before. A plugin: its
       ``Params.from_legacy(config)``, which may map the flat ``z_*`` fields.

``Params.from_legacy`` returns the class defaults unless the plugin overrides it, so a
plugin stage selected by a flat field runs with its defaults; to pass other values, build
the pipeline with :ref:`presets.track <pipeline-presets>` (``relax=("acme.taubin",
{"passes": 20})``) and hand the spec to ``TrackGenerator``. The width and validity
stages have no flat field: use the preset (``width=...``, ``checks=(...)``).

The names are resolved when the generator is constructed, in the order
``relax_solver``, ``smooth_finish``, ``z_profile``, after the generator and its
``generator_params``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Cause
     - Error at ``TrackGenerator`` construction
   * - ``relax_solver="energy"`` or ``"tp_sobolev"``
     - ``ValueError``: an oracle-only backend (``tests/_oracle``), not a relax stage; the
       message lists the registered relax stages.
   * - Another unknown ``relax_solver``
     - ``ValueError``: ``TrackGenConfig.relax_solver: unknown relax 'bogus'; available:
       ['none', 'xpbd']`` (plus the installed plugins' names).
   * - An unknown ``z_profile``
     - ``ValueError``: ``TrackGenConfig.z_profile: unknown elevation 'bogus';
       available: ['flat', 'noise', 'random_walk', 'uniform']`` (plus the installed
       plugins' names).
   * - A name served by a plugin that failed to load
     - That plugin's ``PluginLoadError``.

``TrackGenConfig`` construction no longer checks ``z_profile`` against a fixed set: it
only requires a non-empty string, since the registry, not the config, knows the
installed stages. ``TrackGenConfig(z_profile="bogus")`` therefore constructs, and
``TrackGenerator`` raises the error above; the ``z_*`` range rules
(``z_min <= z_max``, ...) still run at config construction. ``GateGenConfig.z_profile``
keeps its four names, checked at config construction, until the gate pipeline opens
(roadmap PR11). ``Course`` takes a ``TrackGenConfig``, so both fields work there too.

.. _generator-params-config:

Per-generator parameters: ``generator_params``
----------------------------------------------

Every generator can declare its parameters as a Params class (see
:ref:`the generator guide <generator-params>`). The last field of both configs,
``generator_params``, selects the values:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``generator_params``
     - Meaning
   * - ``None`` (default)
     - The generator's ``Params.from_legacy(config)``. For a built-in, that copies the flat
       fields listed below (``polar_num_knots``, ``voronoi_control_points``, ...), so every
       existing config keeps its meaning and its outputs, byte for byte. A plugin gets its
       class defaults unless it overrides ``from_legacy``.
   * - A mapping
     - The generator's complete parameter set, by field name. Keys left out take the
       class defaults, not the flat fields. An unknown key raises ``ValueError`` listing
       the valid fields.
   * - A Params instance
     - The same, as an instance of the generator's own class. An instance of another
       class raises ``TypeError`` naming the expected one.

These three configs build identical generators:

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.plugin import registry

   PolarParams = registry.get("generator", "polar").params

   TrackGenConfig(generator="polar", polar_num_knots=9)
   TrackGenConfig(generator="polar", generator_params={"polar_num_knots": 9})
   TrackGenConfig(generator="polar", generator_params=PolarParams(polar_num_knots=9))

Rules:

- **Set a knob in one place.** With ``generator_params`` given, a flat field that the
  selected built-in reads must stay at its dataclass default or equal the
  ``generator_params`` value; otherwise construction raises ``ValueError`` naming the
  fields. The flat fields of the other generators are ignored, as before.
- **Switching generators means resetting it.** ``dataclasses.replace(cfg,
  generator="polar")`` on a config whose ``generator_params`` is a ``BezierParams`` raises
  ``TypeError``; pass ``generator_params=None`` (or polar's) in the same ``replace``.
- **It is resolved when the generator is constructed** (``TrackGenerator``,
  ``GateGenerator``, hence ``Course``), together with the rules of the selected generator
  (see :ref:`generator-params-validation`). Config construction checks only its type: a
  value that is not ``None``, a mapping or a dataclass instance raises ``TypeError``.
- **A generator without a Params class takes none.** A non-empty mapping raises
  ``ValueError`` ("takes no generator_params").
- ``gen.config`` returns the config as passed, with a mapping normalised to the Params
  instance, so ``TrackGenerator(gen.config, rng)`` builds an identical generator.
  ``gen.params["generator"].params`` is the resolved instance, also when
  ``generator_params`` is ``None``. Both hold the construction-time (initial) values of a
  Stage's *dynamic* parameters: after ``gen.params["generator"].set("amp", ...)`` they
  still report the initial value, and a generator rebuilt from ``gen.config`` starts
  from it. The current values are the live arrays, ``gen.params["generator"]["amp"]``.
- A ``Course`` passes ``generator_params`` through:
  ``CourseConfig(gen=TrackGenConfig(..., generator_params=...))``.
- ``GateGenConfig.generator_params`` works the same way, and the gate-count check
  (``min_gates <= max_gates(config) <= config.max_gates``) sees the resolved values:
  ``GateGenConfig(generator="polar", generator_params={"polar_num_knots": 9})`` makes 9
  gates.

The built-in Params classes have no public import path: pass a mapping, or reach a class
through ``track_gen.plugin.registry.get(kind, name).params`` and its schema through
``registry.describe(kind, name).params_schema``. Their field names are the flat ones,
prefixes included (``polar_num_knots``, ``voronoi_control_points``, ``repulsive_*``), and
they are the built-ins' permanent parameter names through 1.0: ``generator_params``
mappings and YAML files written with them stay valid. ``generator``, ``num_envs``, ``device``,
``num_points``, ``scale`` and every relaxation, inflation, spacing and Z field are core
fields: they stay flat and belong to no Params class. A Stage generator receives ``scale``
as ``StageBuilder.scale``, and ``num_points`` is its default raw point count.

.. _builtin-generator-params:

Built-in generator parameters
-----------------------------

Every field is static, its default is the config field's default, and the rule in the last
column is checked when the generator is constructed, with the same message as before
(``lo``/``hi`` in the schema mirror these rules). A field without a rule is used as given;
where the generator clamps a value, the description says so. The descriptions of each knob
are on the generator pages (:doc:`/generators/overview`).

Track generators
~~~~~~~~~~~~~~~~

``BezierParams`` (``generator="bezier"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``min_num_points``
     - ``9``
     - ``>= 2``
   * - ``max_num_points``
     - ``13``
     - ``>= min_num_points``
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``min_point_distance``
     - ``0.05``
     - in ``(0, 0.5]``
   * - ``rad``
     - ``0.4``
     -
   * - ``edgy``
     - ``0.0``
     -
   * - ``handle_clamp_frac``
     - ``0.4``
     -
   * - ``style_sampling``
     - ``False``
     -
   * - ``rad_range``
     - ``None``
     - ``(lo, hi)`` or ``None``
   * - ``scale_range``
     - ``None``
     - ``(lo, hi)`` or ``None``
   * - ``handle_clamp_frac_range``
     - ``None``
     - ``(lo, hi)`` or ``None``

``HullParams`` (``generator="hull"``): ``min_num_points``, ``max_num_points``,
``num_points_per_segment`` and ``min_point_distance`` with the defaults and rules of
``bezier`` (the two share the corner sampler), plus:

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``hull_displacement``
     - ``0.15``
     -

``PolarParams`` (``generator="polar"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``polar_num_knots``
     - ``12``
     - none (the generator clamps it to ``>= 4``)
   * - ``polar_radial_jitter``
     - ``0.60``
     - none (clamped to ``[0, 0.85]``)
   * - ``polar_angular_jitter``
     - ``0.30``
     - none (clamped to ``[0, 0.45]``)

``VoronoiParams`` (``generator="voronoi"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``voronoi_num_sites``
     - ``256``
     - ``>= voronoi_control_points``
   * - ``voronoi_site_layout``
     - ``"void_ring"``
     - one of ``"ring"``, ``"void_ring"``, ``"clustered"``, ``"mixed"``
   * - ``voronoi_control_points``
     - ``18``
     - ``>= 6``
   * - ``voronoi_radial_variation``
     - ``0.62``
     - ``>= 0``
   * - ``voronoi_angular_jitter``
     - ``0.08``
     - ``>= 0``

``CheckpointParams`` (``generator="checkpoint"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``checkpoint_count``
     - ``12``
     - ``>= 3``
   * - ``checkpoint_radius_min_frac``
     - ``0.33``
     - in ``[0, 1)``
   * - ``checkpoint_angle_jitter``
     - ``0.55``
     -
   * - ``checkpoint_turn_rate``
     - ``0.42``
     -
   * - ``checkpoint_steer_gain``
     - ``0.65``
     -
   * - ``checkpoint_lookahead_frac``
     - ``0.16``
     -
   * - ``checkpoint_best_of_k``
     - ``4``
     - ``>= 1``
   * - ``checkpoint_clip_fallback``
     - ``False``
     -

``RepulsiveParams`` (``generator="repulsive"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``repulsive_grow_mult_min``
     - ``4.5``
     - ``>= 1``
   * - ``repulsive_grow_mult_max``
     - ``5.5``
     - ``> 1`` and ``>= repulsive_grow_mult_min``
   * - ``repulsive_domain_frac``
     - ``0.35``
     - ``> 0``
   * - ``repulsive_domain_init_ratio``
     - ``4.0``
     - ``> 1``
   * - ``repulsive_obstacle_count_min``
     - ``8``
     - ``>= 1``
   * - ``repulsive_obstacle_count_max``
     - ``12``
     - ``>= repulsive_obstacle_count_min``
   * - ``repulsive_obstacle_radius_min_frac``
     - ``0.02``
     - ``> 0``
   * - ``repulsive_obstacle_radius_max_frac``
     - ``0.045``
     - ``>= repulsive_obstacle_radius_min_frac``
   * - ``repulsive_ratchet_rate``
     - ``0.012``
     - ``> 0``
   * - ``repulsive_alpha``
     - ``3.0``
     - ``> 0``
   * - ``repulsive_beta``
     - ``6.0``
     - ``> 0``
   * - ``repulsive_tau``
     - ``0.4``
     -
   * - ``repulsive_w_len``
     - ``30.0``
     -
   * - ``repulsive_stages``
     - ``(64, 128, 256)``
     - non-empty, positive multiples of 4, strictly increasing; the last equals
       ``num_points``
   * - ``repulsive_settle_iters``
     - ``40``
     - ``>= 1``
   * - ``repulsive_resample_every``
     - ``25``
     - ``>= 1``
   * - ``repulsive_stall_window``
     - ``16``
     - ``>= 1``
   * - ``repulsive_stall_area_tol``
     - ``0.05``
     -
   * - ``repulsive_deactivate_obstacles``
     - ``True``
     -

Gate generators
~~~~~~~~~~~~~~~

``PointGateParams`` (``generator="bezier"`` and ``generator="hull"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``min_num_points``
     - ``9``
     - ``>= 2``
   * - ``max_num_points``
     - ``13``
     - ``>= min_num_points``
   * - ``min_point_distance``
     - ``0.05``
     - ``> 0``

``PolarGateParams`` (``generator="polar"``; one gate per knot):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``polar_num_knots``
     - ``12``
     -
   * - ``polar_radial_jitter``
     - ``0.60``
     -
   * - ``polar_angular_jitter``
     - ``0.30``
     -

``VoronoiGateParams`` (``generator="voronoi"``; anchors are emitted directly as gates, so
three control points suffice):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``voronoi_num_sites``
     - ``256``
     - ``>= voronoi_control_points``
   * - ``voronoi_site_layout``
     - ``"void_ring"``
     - one of ``"ring"``, ``"void_ring"``, ``"clustered"``, ``"mixed"``
   * - ``voronoi_control_points``
     - ``18``
     - ``>= 3``
   * - ``voronoi_radial_variation``
     - ``0.62``
     - ``>= 0``
   * - ``voronoi_angular_jitter``
     - ``0.08``
     - ``>= 0``

``CheckpointGateParams`` (``generator="checkpoint"``; one gate per checkpoint):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``checkpoint_count``
     - ``12``
     - ``>= 3``
   * - ``checkpoint_radius_min_frac``
     - ``0.33``
     - in ``[0, 1)``
   * - ``checkpoint_angle_jitter``
     - ``0.55``
     -

The ``GateGenConfig`` fields ``num_points_per_segment``, ``rad``, ``edgy``,
``handle_clamp_frac`` and ``hull_displacement`` belong to no gate generator: they are
inert, and ``generator_params`` cannot set them.

.. _generator-params-validation:

When the rules are checked
--------------------------

Config construction checks the core fields only (for tracks ``num_envs``,
``half_width``, the relaxation and smoothing knobs, ``output_mode``, spacing and the Z
fields, with ``z_profile`` only required to be a non-empty string; for gates
``num_envs``, the gate counts, radius, width, solver iterations, ordering, alignment and
the Z profile) and the type of ``generator_params``. The ``relax_solver`` and
``z_profile`` names are resolved by ``TrackGenerator`` (see
:ref:`flat-stage-shortcuts`). The
generator-specific rules in the tables above run when the generator is constructed
(``TrackGenerator``, ``GateGenerator``, and so ``Course``), and only for the selected
generator: first its ``validate()``, with the same conditions and messages as in
0.2.0.dev1, then the generic ``lo``/``hi``/``choices`` checks. In 0.2.0.dev1 every config
ran every rule in its ``__post_init__``. What follows from the move:

- **An invalid knob of a generator that is not selected is accepted and ignored.**
  ``TrackGenConfig(generator="bezier", voronoi_control_points=3)`` used to raise; it now
  builds, and only ``generator="voronoi"`` rejects it.
- **The errors move from the config to the generator.**
  ``TrackGenConfig(generator="voronoi", voronoi_control_points=3)`` constructs, and
  ``TrackGenerator(config, rng)`` raises the ``ValueError`` the config used to raise.
- **An unknown generator name is reported first**, before any parameter error.
- **The facades' own checks come before the rules.** ``TrackGenerator`` and
  ``GateGenerator`` check ``rng`` (``None``, or a seed count other than ``num_envs``)
  first, and ``Course`` refuses a capture (``CaptureError``) and allocates its rng
  first. With several errors, these are reported before the parameter error: for example
  a ``Course`` built with a bad generator knob inside a capture raises ``CaptureError``,
  where the config used to raise ``ValueError``.
- **The first error can differ when several rules fail.** The rules run in the order of the
  selected generator's ``validate()``. For ``repulsive``, the ``repulsive_stages[-1] ==
  num_points`` rule (which only ever applied to ``generator="repulsive"``) now runs after
  the ``settle_iters``, ``resample_every`` and ``stall_window`` rules.
- **Private helpers do not validate.** Calling ``spec.alloc_scratch(cfg)``,
  ``warp_pipeline._inflate_warp_alloc`` or ``_run_pipeline`` directly skips the
  generator-specific rules.
- **The parameter explorer reports these errors when it generates** (``generate_batch``)
  rather than when it builds the config.

The moved rules are exactly those in the tables: the point-family rules of ``bezier`` and
``hull`` (track and gate), ``num_points_per_segment >= 2`` for ``bezier``, ``hull``,
``polar`` and ``voronoi``, and every ``voronoi_*``, ``repulsive_*`` and ``checkpoint_*``
rule. The new errors, all at generator construction unless noted:

.. list-table::
   :header-rows: 1
   :widths: 56 44

   * - Cause
     - Error
   * - An unknown key in a ``generator_params`` mapping
     - ``ValueError`` listing the valid fields
   * - A non-empty ``generator_params`` for a generator without a Params class
     - ``ValueError`` ("takes no generator_params")
   * - A Params instance of another class
     - ``TypeError`` naming the expected class
   * - A flat field and ``generator_params`` setting the selected built-in's knob to
       different values
     - ``ValueError`` naming the fields
   * - A value outside a field's ``lo``/``hi`` or ``choices``
     - ``ValueError``
   * - A ``generator_params`` that is not ``None``, a mapping or a dataclass instance
     - ``TypeError``, at config construction

.. _deprecated-config-fields:

Deprecated fields
-----------------

``TrackGenerator`` does not use the ``TrackGenConfig`` fields below (it only rejects
``smooth_finish=True``); only the torch oracle in the tests, a benchmark or
``track_gen._experimental.fourier`` read them. Setting any of them to a value other than
its default (the default of the config's own class, for a subclass) emits one
``DeprecationWarning`` at config construction, attributed to the code that constructs the
config and naming every such field: ``TrackGenConfig fields [...] are deprecated and
unused by TrackGenerator (oracle/experimental only; smooth_finish=True is rejected); they
will be removed in 1.0``. Nothing is removed in 0.x, and the fields keep their defaults.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Fields
     - Read only by
   * - ``energy_steps``, ``energy_lr``, ``energy_w_sep``, ``energy_w_len``,
       ``energy_w_bend``, ``energy_w_anchor``
     - The torch oracle and ``benchmarks/benchmark_relaxation.py``
   * - ``tp_iters``, ``tp_tau``, ``tp_alpha``, ``tp_beta``
     - The torch oracle
   * - ``smooth_finish``, ``smooth_finish_iters``, ``smooth_finish_tau``
     - The torch oracle. ``TrackGenerator`` still raises ``ValueError`` for
       ``smooth_finish=True``.
   * - ``relax_chunk_size``, ``relax_use_warp``
     - The torch oracle
   * - ``max_regen_iters``
     - The torch oracle's accept-first-valid loop
   * - ``min_angle``
     - The corner-angle kernel, launched only by the tests
   * - ``num_harmonics``, ``decay_p``, ``amplitude``, ``num_centerline_samples``
     - ``track_gen._experimental.fourier``
   * - ``output_mode``
     - Nothing. It accepts only ``"constant_spacing"`` (any other value raises
       ``ValueError``), so it never warns: it is deprecated in this documentation only.

The inert ``GateGenConfig`` fields listed above are not deprecated in this release.
