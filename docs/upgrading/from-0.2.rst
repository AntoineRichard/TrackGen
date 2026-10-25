.. _upgrading-0.3:

Upgrading from 0.2
==================

The current milestone, ``0.3.0.dev2``, includes the API removals from
``0.3.0.dev1`` and switches every built-in random stream to hashed salts. Generated
track and gate bytes change for the same seed. There is no compatibility shim,
legacy RNG config field or reproduction mode. Poison defaults, frozen configs and
required result fields remain pending for a later milestone.

Update imports and configuration dictionaries, rebaseline stored generated outputs,
run your tests, then review any saved pipeline specs. The :doc:`from-0.1` guide
records the earlier 0.1 to 0.2 transition; examples marked historical there describe
those releases rather than the current API.

.. _upgrading-0.3-rng:

Built-in RNG streams change
---------------------------

Previously, built-in streams initialized Warp's RNG from ``seed * salt + offset``
in int32 arithmetic. At seed 0 the salt disappeared, so streams with the same
offset coincided. Built-ins now mix the effective seed with a fixed named salt
through :mod:`track_gen.ops.rng`. This removes that systematic seed-0 collapse;
finite RNG states still permit collisions.

All six track generators and all five gate generators use hashed keying, as do
``order.random_pairs`` and the nonflat ``uniform``, ``random_walk`` and ``noise``
elevations. Gate generators share the matching track generator's stream names.
The :ref:`fixed stream table <ops-rng-keying>` defines the salts and index meanings;
renaming or moving a built-in Stage does not change its salt. The effective-seed
and epoch formulas, ``PerEnvSeededRNG`` and ``randomize.uniform`` are unchanged.
Third-party plugin kernels retain their own RNG choices.

Rebuild byte goldens and cached datasets once. The same seed remains reproducible
within a fixed device, Warp version and configuration, but it does not reproduce
0.2 geometry. No ``rng="0.2"`` / ``rng="1.0"`` setting exists on configs, presets
or pipeline specs. The ``rng=`` object argument on generator facades still means a
``PerEnvSeededRNG`` seed source, not a choice of keying.

Public ``ops.sample_counts`` and ``ops.sample_grid_points`` keep
``keying="linear"`` as their default so existing plugin callers keep their draws.
Opt in with the keyword-only ``keying="hashed"`` to use
``rng.init_at(seed, salt, offset)``. New plugin kernels can call ``rng.init`` or
``rng.init_at`` directly; their formulas are frozen for plugin API version 1.

A ``Course`` that derives checkpoint capacity by generating geometry at construction
can derive a different capacity after this switch, especially for elevated tracks or
plugin geometry without an analytic length bound. Record the old capacity as
``course.checkpoint_sampler.max_checkpoints`` and pass it explicitly as
``max_checkpoints=`` when constructing the replacement. Choose a larger explicit
capacity if the new geometry needs more checkpoints.

The RNG validation gate compares empirical distributions rather than old track bytes:
all built-in track and gate generators, including ``repulsive``, run on two seed
blocks of at least 4096 envs each. It includes nonzero uniform/walk/noise elevations
and ``order.random_pairs``: 19 cases and 133 distributions. Validity uses a
six-sigma binomial band plus one percentage point. Counts, length, compactness,
curvature, mean altitude and altitude standard deviation use a two-sample empirical
CDF bound with family-wise alpha ``0.001``. The tool reports the observed distances
and limits rather than assuming a pass:

.. code-block:: bash

   # Run capture with each checkout's PYTHONPATH on the same device.
   python tests/tools/rng_statistics.py capture old.npz --device cuda:0
   python tests/tools/rng_statistics.py capture new.npz --device cuda:0
   python tests/tools/rng_statistics.py compare old.npz new.npz

The first capture uses the old-keying checkout; the second uses the new one.
Exact-byte tests are rebaselined for the new streams and continue guarding future
unintended changes. Earlier measured figures and tuning tables in the generator
catalogue remain reference measurements; they are not exact expected values for the
new seed mapping.

.. _upgrading-0.3-removals:

Removed APIs and replacements
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 47 53

   * - Removed
     - Replacement
   * - ``track_gen.set_capturing``, ``track_gen.course.set_capturing``,
       ``Course.set_capturing`` and private ``_CAPTURING`` flags
     - Delete the calls and flag writes. Calls use their launch context and can be
       recorded without a global capture switch.
   * - ``CourseLine.track``
     - ``CourseLine.centerline``
   * - Built-in generator class ``.params`` aliases
     - ``registry.get("generator", name).Params``. Runtime ``gen.params`` and
       ``GeneratorSpec(params=...)`` remain supported.
   * - ``style_sampling``, ``rad_range``, ``scale_range``,
       ``handle_clamp_frac_range`` in configs, Params and saved specs
     - Remove the keys and use ``randomize=``; see the recipe below.
   * - ``output_mode``
     - Delete the key. The config path uses constant-spacing resampling.
   * - Unused ``TrackGenConfig`` fields listed below
     - Delete the keys; experiment-specific parameters belong to your own config.
   * - Inert ``GateGenConfig.num_points_per_segment``, ``rad``, ``edgy``,
       ``handle_clamp_frac`` and ``hull_displacement``
     - Delete the keys. These fields never affected gate anchors.
   * - ``validity.gate_default`` params ``gate_width`` and ``check_crossing``
     - Delete both keys. Set physical bar width on ``gate_frames.default``;
       nonzero gate bars are checked per env.
   * - ``tools.Progress(checkpoints=...)``
     - ``tools.Progress(source=...)``
   * - Ignored ``capturing=`` on private relaxation helpers
     - Delete the argument and pass the launch context.
   * - ``track_gen._src.generator_registry`` and ``gate_generator_registry``;
       ``GENERATORS`` / ``GATE_GENERATORS`` views
     - ``track_gen.plugin.registry``: ``add``, ``get``, ``available``, ``describe``.
   * - ``GeneratorSpec(accepts_ctx=...)`` and context-free generate callables
     - Remove the flag; define ``generate(..., *, ctx)`` and emit device work
       through ``ctx``. The five-argument adapter is gone.
   * - ``LegacyPluginStreamError``, ``legacy_stream`` introspection and the
       ``GraphRunner(legacy_stream=...)`` option
     - Delete legacy-stream branches. Every supported generator uses the call's
       launch context and stream; the shared legacy capture lock is removed.
   * - ``GateGeneratorSpec`` and its adapter/staging object
     - A ``Stage`` with ``kind="gate_generator"`` and ``output="anchors"``;
       see :ref:`writing-a-gate-generator`.
   * - ``warp_gate.alloc_order_scratch``, ``order_points``, ``finish_ordered_gates``
     - The pipeline's order and normalize stages; ``track_gen.ops`` for shared math.
   * - ``warp_pipeline`` generation re-exports, ``GenScratch`` compatibility name,
       private helper aliases and module hooks
     - Import public ops directly; see :ref:`legacy-src-helpers`.
   * - Frozen ``generate_*_warp`` and ``*_alloc_scratch`` functions,
       ``apply_z_profile*`` and scalar sampling helpers
     - Use ``TrackGenerator``, ``GateGenerator`` or a pipeline preset. Their scalar
       and style kernels are removed too; live private Stage implementations remain.
   * - ``track_gen.pipeline/1`` dictionaries
     - Load and re-save as ``/2`` with TrackGen 0.2 before upgrading. Current
       ``PipelineSpec.from_dict`` refuses ``/1``.

The unused track fields are ``energy_steps``, ``energy_lr``, ``energy_w_sep``,
``energy_w_len``, ``energy_w_bend``, ``energy_w_anchor``, ``tp_iters``, ``tp_tau``,
``tp_alpha``, ``tp_beta``, ``smooth_finish``, ``smooth_finish_iters``,
``smooth_finish_tau``, ``relax_chunk_size``, ``relax_use_warp``, ``max_regen_iters``,
``min_angle``, ``num_harmonics``, ``decay_p``, ``amplitude`` and
``num_centerline_samples``. They are absent from dataclass fields, ``asdict`` and
constructor signatures. Passing them raises ``TypeError``; there are no tombstones.
Remove them from YAML/Hydra dumps and dictionaries as well as Python calls.

Per-env randomization
---------------------

A former ``scale_range=(a, b)`` becomes ``generator.size=(a / scale, b / scale)``;
``rad_range`` becomes ``generator.rad`` and ``handle_clamp_frac_range`` becomes
``generator.handle_clamp_frac``. The size range must lie in ``[0.1, 10]``.
Randomizers preserve the requested uniform distributions, with different draws from
the former style kernel; they do not reproduce a removed style configuration's bytes.

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       num_envs=4, device="cpu",
       randomize={"generator.size": (0.8, 1.2),
                  "generator.rad": (0.1, 0.35),
                  "generator.handle_clamp_frac": (0.3, 0.9)}))

For a config, convert the cleaned config with
``legacy.to_pipeline_spec(cfg).with_randomize({...})``. A course accepts that spec
through ``Course.track(spec, checkpoint_spacing=...)``. A curriculum can also write
``course.generator.params["generator"].set("size", values)`` before regenerating.

Plugins
-------

``GeneratorSpec`` remains the simple track-plugin path because a fixed-shape
static generator needs only an allocator and a generate callable. It uses the same
pipeline and conformance kit as a Stage. Import it from ``track_gen.plugin``;
``generate`` must accept the required ``ctx=`` keyword. The ``accepts_ctx``
constructor flag is removed, and every launch, copy and fill goes through the context.
Use a Stage for dynamic parameters, attributes and bounded device loops.

Gate plugins use Stages. Move allocation to ``build`` and ``b.scratch``, write
``anchors.xy`` / ``anchors.count`` with ``b.write``, declare capacity, and let the
pipeline order and normalize the anchors. To preserve a plugin's already ordered
gates, a Stage can instead write ``gates.xy`` / ``gates.count`` and declare
``provides={"order"}``; no order or normalize stage follows it. Keep the old kernels
and RNG constants when preserving a plugin's bytes. Run ``check_generator`` or
``check_gate_generator`` after the port (:ref:`testing-a-plugin`).

Later 0.3 milestones
--------------------

The approved later work changes every entry point's invalid-output default to
``on_invalid="poison"``, freezes config dataclasses and makes the newer Track and
GateSequence fields required. These are pending changes, not behavior in
``0.3.0.dev2``; they will be documented here when they land.
The final release follows an rc tested by the downstream racing repositories.
