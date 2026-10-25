.. _upgrading-0.3:

Upgrading from 0.2
==================

The 0.3 development line removes obsolete APIs without compatibility shims or
another warning cycle. This page currently describes the removal milestone
``0.3.0.dev1``. Surviving configurations keep their generated geometry at this
milestone. The hashed RNG switch, poison defaults, frozen configs and required
result fields are later milestones; they are not active in this removal milestone.

Update imports and configuration dictionaries first, run your tests, then review
any stored pipeline specs. There is no compatibility option for a removed name.
The :doc:`from-0.1` guide records the earlier 0.1 to 0.2 transition; examples marked
historical there describe those releases rather than the current API.

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

The approved later work switches built-in streams to hashed salts with no legacy
reproduction mode, changes every entry point's invalid-output default to
``on_invalid="poison"``, freezes config dataclasses and makes the newer Track and
GateSequence fields required. Those changes will be documented here when they land.
The final release follows an rc tested by the downstream racing repositories.
