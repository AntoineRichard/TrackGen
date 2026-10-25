.. _reference-testing:

Conformance kit
===============

The checks a generator, :term:`stage` or :term:`component` plugin runs in its own CI,
and TrackGen runs over every built-in; see :ref:`testing-a-plugin`. ``track_gen.testing``
is a subpackage outside ``track_gen.__all__``: import it explicitly. ``import
track_gen`` does not import it, and importing it imports neither torch nor pytest.

.. automodule:: track_gen.testing
   :no-members:

Conformance checks
------------------

.. automodule:: track_gen.testing.conformance
   :no-members:

.. autofunction:: track_gen.testing.check_generator

.. autofunction:: track_gen.testing.check_gate_generator

.. autofunction:: track_gen.testing.check_stage

.. py:currentmodule:: track_gen.testing

.. py:data:: CHECKS
   :type: Mapping[str, tuple[str, ...]]

   The check names of each registry kind (``"generator"``, ``"gate_generator"``,
   ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"``,
   ``"stations"``) and of ``"component"`` (:func:`check_component`), in run order. A
   read-only mapping.

.. py:data:: CHECKS
   :module: track_gen.testing.conformance

   The same mapping as :data:`track_gen.testing.CHECKS`.

.. autoclass:: track_gen.testing.Report
   :members: ok, failures, failed_names, result, raise_for_failures

   ``kind`` and ``name`` identify the generator or stage; ``results`` is the list of
   :class:`CheckResult <track_gen.testing.CheckResult>`, one per check and device.

.. autoclass:: track_gen.testing.CheckResult
   :no-members:

   .. autoproperty:: status

Component checks
----------------

.. automodule:: track_gen.testing.components
   :no-members:

.. autofunction:: track_gen.testing.check_component

Hot-path guards
---------------

.. automodule:: track_gen.testing.contract
   :no-members:

.. autofunction:: track_gen.testing.hot_path_guard

.. autofunction:: track_gen.testing.no_host_sync

.. autofunction:: track_gen.testing.no_allocation

.. autofunction:: track_gen.testing.stream_explicit

The four functions above are the entry points; the class below is the type they return.

.. autoclass:: track_gen.testing.HotPathGuard
   :members: ok, of, allocated, report, raise_for_violations

.. autoclass:: track_gen.testing.Violation
   :no-members:

.. autoclass:: track_gen.testing.Site
   :no-members:

.. py:currentmodule:: track_gen.testing

.. py:data:: CATEGORIES
   :type: tuple[str, ...]
   :value: ("host_sync", "allocation", "stream", "capture")

   The :class:`~track_gen.testing.Violation` categories, in report order (the argument
   of :meth:`HotPathGuard.of() <track_gen.testing.HotPathGuard.of>`).

.. py:data:: CATEGORIES
   :module: track_gen.testing.contract

   The same tuple as :data:`track_gen.testing.CATEGORIES`.

.. autoexception:: track_gen.testing.HotPathViolation

Launch traces
-------------

.. automodule:: track_gen.testing.trace
   :no-members:

.. autofunction:: track_gen.testing.trace_launches

The type :func:`~track_gen.testing.trace_launches` returns:

.. autoclass:: track_gen.testing.LaunchTrace
   :members: kernels, topology, diff, same_topology, foreign_pointers

.. autoclass:: track_gen.testing.LaunchRecord
   :no-members:

.. autofunction:: track_gen.testing.buffers_of

.. autoclass:: track_gen.testing.BufferSet
   :members: add, locate, labels

.. autofunction:: track_gen.testing.assert_same_topology

.. autofunction:: track_gen.testing.assert_pointers_within

Caller capture (CUDA)
---------------------

.. automodule:: track_gen.testing.capture
   :no-members:

.. autofunction:: track_gen.testing.assert_caller_capturable

.. py:currentmodule:: track_gen.testing

.. py:data:: CAPTURE_MODES
   :type: tuple[str, ...]
   :value: ("warp", "external", "torch")

   Every :term:`capture` mode :func:`~track_gen.testing.assert_caller_capturable`
   knows, in run order.

.. py:data:: CAPTURE_MODES
   :module: track_gen.testing.capture

   The same tuple as :data:`track_gen.testing.CAPTURE_MODES`.

The building blocks of :func:`~track_gen.testing.assert_caller_capturable`, for custom
capture tests (import them from ``track_gen.testing.capture``):

.. autofunction:: track_gen.testing.capture.side_stream

.. autofunction:: track_gen.testing.capture.mempool_disabled

.. autofunction:: track_gen.testing.capture.capture_launch_forbidden

.. autofunction:: track_gen.testing.capture.recording

.. autofunction:: track_gen.testing.capture.poison

.. autofunction:: track_gen.testing.capture.poison_all

.. autofunction:: track_gen.testing.capture.result_arrays

.. py:currentmodule:: None

pytest plugin
-------------

``track_gen.testing.pytest_plugin`` is opt-in (there is no ``pytest11`` entry point):
load it with ``pytest_plugins = ["track_gen.testing.pytest_plugin"]`` in a top-level
``conftest.py``, or ``-p track_gen.testing.pytest_plugin``. It imports pytest, so this
reference documents it by hand rather than importing it.

.. py:module:: track_gen.testing.pytest_plugin

.. py:currentmodule:: track_gen.testing.pytest_plugin

.. py:function:: track_gen_conformance(name, params=None, *, kind="generator", **kwargs)

   A pytest fixture whose value is this factory. It runs
   :func:`~track_gen.testing.check_generator` (with ``kind="gate_generator"``,
   :func:`~track_gen.testing.check_gate_generator`; with ``kind`` one of ``"relax"``,
   ``"width"``, ``"elevation"``, ``"validity"``, ``"order"``, ``"stations"``,
   :func:`~track_gen.testing.check_stage`)
   on ``name`` with ``params`` and ``kwargs``, fails the test listing every failed check
   unless all pass, and returns the :class:`~track_gen.testing.Report`.

.. py:function:: track_gen_check_component(factory, **kwargs)

   A pytest fixture whose value is this factory. It runs
   :func:`~track_gen.testing.check_component` on ``factory`` with ``kwargs``, fails the
   test listing every failed check unless all pass, and returns the
   :class:`~track_gen.testing.Report`.

.. py:function:: generator_names(kind="generator", *, builtin=None, exclude=()) -> list[str]

   The names registered under ``kind`` (``"generator"``, ``"gate_generator"`` or one of
   the six stage kinds), sorted, without those in ``exclude``. ``builtin=True`` keeps
   only the built-ins, ``builtin=False`` only the third-party ones.

.. py:function:: parametrize_generators(kind="generator", *, argname="generator", builtin=None, exclude=())

   ``pytest.mark.parametrize(argname, generator_names(kind, builtin=builtin,
   exclude=exclude))``. It reads the registry at collection time, so the plugins its
   entry points name load then.

.. py:currentmodule:: None
