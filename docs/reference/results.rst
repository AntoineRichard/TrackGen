.. _reference-results:

Results
=======

The result objects a generator writes, their ``Layout``, the capability protocols the
runtime utilities accept, and the check they run; :ref:`results` explains how to read
them. Every result holds per-:term:`env` arrays (``[E]``, such as :term:`valid`) and
per-point arrays (``[E * stride]``). ``Track``, ``GateSequence``, ``Layout`` and the
protocols are also importable from ``track_gen``, and ``track_gen.plugin.Layout`` is the
same class as ``track_gen.results.Layout``.

.. automodule:: track_gen.results
   :no-members:

.. autoclass:: track_gen.results.Layout
   :no-members:

Tracks and gate sequences
-------------------------

.. autoclass:: track_gen.results.Track
   :no-members:

   .. automethod:: validate

   .. automethod:: view

   .. automethod:: copy_from

   .. automethod:: clone

   .. automethod:: alloc

   ``fail_mask`` and ``fail_bits`` say why an env is invalid (its :term:`fail bits`):
   the bit table is :data:`~track_gen.pipeline.BUILTIN_FAIL_BITS` plus the pipeline's
   plugin checks (see :ref:`pipeline-fail-bits`). ``half_width`` is the band's per-point
   half-width, and ``attributes`` the per-point attributes the pipeline exports (see
   :ref:`pipeline-attributes`). ``layout`` is ``None`` on a ``Track`` built by hand
   (:ref:`results-layout`).

.. autoclass:: track_gen.results.GateSequence
   :no-members:

   .. automethod:: validate

   .. automethod:: view

   .. automethod:: copy_from

   .. automethod:: clone

   .. automethod:: alloc

   ``fail_mask`` and ``fail_bits`` say why an env is invalid: the bit table is
   :data:`~track_gen.pipeline.GATE_FAIL_BITS` plus the pipeline's plugin checks (see
   :ref:`pipeline-gate-fail-bits`). ``fail_mask`` is ``None`` on a ``GateSequence`` built
   elsewhere; when it is set, ``valid[e] == (fail_mask[e] == 0)``. ``attributes`` holds
   the per-gate attributes the pipeline exports (see :ref:`pipeline-gate-plugin-channels`).

.. autoclass:: track_gen.results.Centerline
   :no-members:

   .. automethod:: validate

   .. automethod:: view

   .. automethod:: copy_from

   .. automethod:: clone

   .. automethod:: alloc

Capability protocols
--------------------

What each runtime utility reads from its source (:ref:`results-protocols`). A shipped
protocol never changes: a new capability is a new protocol.

.. autoclass:: track_gen.results.HasCenterline
   :no-members:

.. autoclass:: track_gen.results.HasBand
   :no-members:

.. autoclass:: track_gen.results.HasGates
   :no-members:

.. autoclass:: track_gen.results.HasGateFrames
   :no-members:

.. autoclass:: track_gen.results.HasGeneration
   :no-members:

.. autoclass:: track_gen.results.HasValidity
   :no-members:

.. autoclass:: track_gen.results.HasCheckpoints
   :no-members:

.. autofunction:: track_gen.results.check_source
