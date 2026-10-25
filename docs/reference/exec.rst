.. _reference-exec:

Execution layer
===============

How TrackGen emits device work: stream resolution, the execution policy shared by every
producer (record into your :term:`capture`, eager, or replay of the producer's own CUDA
graph; segment by segment for a pipeline with ``HOST`` stages), the device-side
:term:`conditional node` skip, bounded loops, the host operations of ``HOST`` stages and
the capture errors; then the per-env :term:`seed` and :term:`epoch` state, and the torch
interop. :ref:`generating-calls` and :ref:`capture-rules` show them in use;
:doc:`/how-it-works/cuda-graph` explains the mechanics.

Launch contexts and runners
---------------------------

.. automodule:: track_gen.exec
   :no-members:

.. autoclass:: track_gen.exec.LaunchContext
   :members:

.. autoclass:: track_gen.exec.GraphRunner
   :members:

.. autoclass:: track_gen.exec.HostOps
   :members:

.. autofunction:: track_gen.exec.resolve_stream

.. autofunction:: track_gen.exec.resolve

.. autofunction:: track_gen.exec.is_capturing

.. autofunction:: track_gen.exec.unconditional

.. autofunction:: track_gen.exec.conditional

Capture errors
--------------

Their causes and fixes are summarized in :ref:`api-errors`.

.. autoexception:: track_gen.exec.CaptureError

.. autoexception:: track_gen.exec.NotCapturableError

.. autoexception:: track_gen.exec.RebindAfterCaptureWarning

Seeds and epochs
----------------

Per-env seeds and device-side epochs, the pair that keys every :term:`generation`; see
:ref:`generating-seeds`.

.. automodule:: track_gen.seeds
   :no-members:

.. autoclass:: track_gen.seeds.SeedState
   :members:

.. autofunction:: track_gen.seeds.effective_seed

Torch interop
-------------

Registering a ``torch.cuda.graph`` capture with Warp, and tying producers' lifetime to a
torch graph; see :ref:`cuda-graph-sim-loop`. Importing ``track_gen`` does not import
torch.

.. automodule:: track_gen.interop.torch
   :no-members:

.. autofunction:: track_gen.interop.torch.registered

.. autofunction:: track_gen.interop.torch.retain
