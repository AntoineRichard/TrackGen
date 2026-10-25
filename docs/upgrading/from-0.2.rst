.. _upgrading-0.3:

Upgrading from 0.2
==================

The 0.3 development line removes the old compatibility APIs, switches built-in RNG
streams to hashed salts, defaults invalid results to poison, freezes configs and
requires complete Track and GateSequence results. These are intentional breaks:
there are no compatibility shims, legacy RNG settings or reproduction modes.
``0.3.0.dev1`` covers removals, ``dev2`` RNG, ``dev3`` the defaults and contracts,
and ``dev4`` the completed documentation. The final release still needs an rc tested
by the downstream racing repositories.

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

Public ``ops.sample_counts`` and ``ops.sample_grid_points`` use
``rng.init_at(seed, salt, offset)``. Their ``keying=`` selector is removed:
``salt`` is a hashed uint32 namespace and ``offset`` an int32 stream index.
Their draws differ from 0.2 for the same salt. New plugin kernels can call
``rng.init`` or ``rng.init_at`` directly; their formulas are frozen for plugin
API version 1.

A ``Course`` that derives checkpoint capacity by generating geometry at construction
can derive a different capacity after this switch, especially for elevated tracks or
plugin geometry without an analytic length bound. Record the old capacity as
``course.checkpoint_sampler.max_checkpoints`` and pass it explicitly as
``max_checkpoints=`` when constructing the replacement. Choose a larger explicit
capacity if the new geometry needs more checkpoints.

The RNG distribution gate passed all **133 distributions across 19 cases**, each with
**8192 envs** (two blocks of 4096 seeds) on the same GPU before and after the switch.
It covers every built-in track and gate generator, including ``repulsive``, nonzero
uniform/walk/noise elevations and ``order.random_pairs``. The largest observed
absolute yield change was **0.00146484375** (about 0.1465 percentage points).
Validity uses a six-sigma binomial band plus one percentage point. Counts, length,
compactness, curvature, mean altitude and altitude standard deviation use a
two-sample empirical CDF bound with family-wise alpha ``0.001``. To repeat the check:

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
   * - ``registry.describe(...).accepts_ctx``
     - Removed: every generator receives ``ctx``.
   * - ``CourseLine.track``
     - ``CourseLine.centerline``
   * - Built-in generator class ``.params`` aliases
     - ``registry.get("generator", name).Params``. Runtime ``gen.params`` and
       ``GeneratorSpec(params=...)`` remain supported.
   * - ``style_sampling``, ``rad_range``, ``scale_range``,
       ``handle_clamp_frac_range`` in configs, Params and saved specs
     - Remove the keys and use ``randomize=``; see the recipe below.
   * - ``ops.sample_counts(..., keying=...)`` and
       ``ops.sample_grid_points(..., keying=...)``
     - Delete the selector. Salts are hashed uint32 namespaces; draws differ
       from 0.2 for the same salt.
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

.. _upgrading-0.3-poison:

Invalid results are poisoned by default
---------------------------------------

``on_invalid="poison"`` is the default on ``TrackGenConfig``, ``GateGenConfig``,
``BatchSpec``, both presets, and the resulting Program. A saved ``/2`` spec whose
batch omits ``on_invalid`` also loads with ``"poison"``; saving it writes the policy
explicitly. Invalid policy names raise ``ValueError``.

An invalid env now has ``count = 0``, NaN geometry in every slot, attribute padding
(NaN for floats, ``-1`` for integers), and zero track length and winding. Its
``valid`` stays zero; ``fail_mask``, ``truncated`` and ``generation`` stay readable.
Valid envs and unmasked envs in a masked regeneration are unaffected by the choice.
If diagnostics need failed geometry, opt in explicitly:

.. code-block:: python

   from track_gen import TrackGenConfig, TrackGenerator
   from track_gen.pipeline import presets

   cfg = TrackGenConfig(num_envs=4, device="cpu", on_invalid="keep")
   diagnostic = TrackGenerator(cfg)
   spec = presets.gates(num_envs=4, device="cpu", on_invalid="keep")
   assert cfg.on_invalid == spec.batch.on_invalid == "keep"

Read ``valid[e]`` before indexing a result, then check ``count[e]`` before taking a
neighbor modulo the count. Zero is a legitimate empty count. Mask rewards and
observations with a select (``np.where(valid, reward, 0.0)`` or ``torch.where``),
never multiplication: ``NaN * 0`` is NaN. Regenerate invalid envs instead of using
their poisoned geometry (:ref:`invalid-envs`). Runtime utilities keep their existing
invalid-env sentinels under both policies.

.. _upgrading-0.3-frozen:

Configs are frozen values
-------------------------

``TrackGenConfig`` and ``GateGenConfig`` are frozen dataclasses. Attribute assignment
and deletion raise ``dataclasses.FrozenInstanceError``; use ``dataclasses.replace``
and build a new generator for a static change. Generator snapshots are frozen too.
Dynamic curricula continue through ``gen.params``; a config's initial values do not
track later device parameter updates.

``spacing=None`` is resolved when a Track config is constructed. Replacing
``half_width`` alone preserves that resolved spacing. Reset it to ``None`` to derive
``0.6 * half_width`` again:

.. code-block:: python

   import dataclasses
   from track_gen import TrackGenConfig

   cfg = TrackGenConfig(num_envs=4, device="cpu", half_width=0.05)
   wider = dataclasses.replace(cfg, half_width=0.1, spacing=None)
   assert abs(wider.spacing - 0.06) < 1e-12
   assert abs(cfg.spacing - 0.03) < 1e-12

Decorate config subclasses as frozen dataclasses too; Python refuses a non-frozen
dataclass subclass of a frozen base:

.. code-block:: python

   @dataclasses.dataclass(frozen=True)
   class MyTrackConfig(TrackGenConfig):
       experiment_label: str = "baseline"

   custom = dataclasses.replace(MyTrackConfig(device="cpu"), experiment_label="wide")
   assert custom.experiment_label == "wide"

This is dataclass freezing, not recursive freezing of objects a user puts inside a
mapping. Do not mutate a nested parameter mapping as a way to reconfigure a live
generator; use a replacement config or the supported dynamic parameter interface.

.. _upgrading-0.3-results:

Track and GateSequence require complete storage
------------------------------------------------

``Track.truncated``, ``generation``, ``fail_mask``, ``half_width`` and ``layout`` are
required constructor arguments. ``GateSequence.generation``, ``fail_mask`` and
``layout`` are required too. All concrete Track/GateSequence array fields and
``layout`` reject explicit ``None`` with ``ValueError``; omitting a required argument
raises ``TypeError``. ``validate()`` also rejects a required field later replaced with
``None``. Full shape, dtype and device checks still happen in ``validate()``, rather
than the constructor.

For manual storage and ``out=`` pipelines, allocate a complete result from a Layout:

.. code-block:: python

   from track_gen.results import GateSequence, Layout, Track

   track = Track.alloc(Layout(4, 384, "cpu"))
   gates = GateSequence.alloc(Layout(4, 16, "cpu"))
   assert track.validate().stride == 384
   assert gates.validate().stride == 16
   assert (track.count.numpy() == 0).all() and (track.valid.numpy() == 0).all()
   assert track.fail_bits == {} and track.attributes == {}

``fail_bits`` and ``attributes`` remain optional metadata arguments with independent
empty-mapping defaults. Pipelines populate their bit names; an allocator does not
invent failure reasons for empty storage. Even a pipeline that provides a Track
without a road offset keeps a required ``half_width`` array, with NaN where no width
is available. ``out=`` no longer fills missing arrays with private buffers or infers
a Track/GateSequence layout.

``Centerline`` keeps its optional ``generation`` and inferred layout. Runtime
capability protocols still accept suitable duck-typed sources and infer their layout
from the members a consumer reads (:ref:`results-check-source`). They do not require
you to construct a Track when your utility only needs a centerline.

Retained arithmetic and release gate
-------------------------------------

The static ``edgy`` parameter is retained. Its ``atan(edgy) / pi + 0.5`` transform
still runs on the host in double precision; making it dynamic would change that
arithmetic. No additional geometry policy or reproduction mode is introduced.

The final 0.3.0 release still requires an rc and validation in the downstream racing
repositories; completing these development milestones does not complete that gate.
