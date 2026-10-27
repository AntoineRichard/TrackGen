.. _extending-trackgen:

Build and extend
================

Start with the smallest extension that supplies what you need. A new track does not
necessarily need a new generator: imported points, a width profile or a gate-station
choice may be enough.

.. list-table:: Choose your route
   :header-rows: 1
   :widths: 31 37 32

   * - You have / want
     - Use
     - Start here
   * - Circuit points from a dataset, editor or learned model
     - Built-in ``external`` generator; load your closed loops
     - :ref:`importing-a-track`
   * - Gates along a track or supplied route
     - ``presets.gates(source="stations")`` or ``"relaxed_stations"``
     - :doc:`/tutorials/gate-sequences`
   * - A new procedural centerline with fixed parameters
     - ``GeneratorSpec``: allocate scratch once, generate through ``ctx``
     - :ref:`generator-spec-recipe`
   * - A generator with live settings, attributes or supplied 3D data
     - A generator ``Stage`` and typed ``Params``
     - :ref:`writing-a-generator`
   * - A new native gate layout or ordering
     - A ``gate_generator`` / ``order`` Stage
     - :ref:`writing-a-gate-generator`
   * - Different width, elevation, relaxation or checks
     - Replace a pipeline Stage
     - :ref:`writing-a-stage`
   * - A new per-step simulator operation
     - A runtime component attached to a ``Course``
     - :ref:`writing-a-component`

.. figure:: ../assets/racing/data-import.png
   :alt: The same supplied loop becomes a road or an ordered sequence of gates.

   Reuse geometry first: the standard pipeline can add road width, resampling,
   attributes, elevation and gates to a loop you already have.

The common contract
-------------------

Allocate and bind stable buffers at construction. During generation or a simulator
step, write in place through the call's launch context; avoid allocations and host
synchronization. Report real counts and validity instead of changing batch membership.
:ref:`results` shows the array layout; :ref:`generator-hard-rules` gives the detailed
kernel/capture rules.

Test an extension with :ref:`the conformance kit <testing-a-plugin>`, then publish it
as a normal Python package with an entry point (:ref:`packaging-a-plugin`). A plugin
uses public modules and needs no edit to TrackGen itself.
