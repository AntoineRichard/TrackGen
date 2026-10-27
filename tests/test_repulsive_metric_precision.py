"""Supported Sobolev metrics must remain positive after storage and application."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from track_gen._src import warp_generate_repulsive as rep
from track_gen.exec import LaunchContext
from tests.conftest import DEVS


_CASES = [(256, 5. / 3.), (512, 1.95), (1024, 1.95)]


def _expected_filter(n, s):
    # Independent stable eigenvalues; avoid cancellation in 2-2*cos(theta).
    k = np.arange(n // 2 + 1, dtype=np.float64)
    eigenvalues = (np.sin(np.pi * k / n) / np.sin(np.pi / n)) ** 2
    result = 1. / (eigenvalues ** s + 1e-3)
    result[0] = 0.
    return result


def _apply(row, points, dev, *, frozen=None, initial=None):
    n = len(row)
    e = len(points) // n
    stored_type = wp.float64 if row.dtype == np.float64 else wp.float32
    h = wp.array(row, dtype=stored_type, device=dev)
    x = wp.array(np.asarray(points, np.float32), dtype=wp.vec2f, device=dev)
    mask = wp.array(np.zeros(e, np.int32) if frozen is None else np.asarray(frozen, np.int32),
                    dtype=wp.int32, device=dev)
    if initial is None:
        out = wp.empty(len(points), dtype=wp.vec2d, device=dev)
    else:
        out = wp.array(np.asarray(initial, np.float64), dtype=wp.vec2d, device=dev)
    LaunchContext.current(dev).launch(rep._conv_k, (e, n), [x, h, n, mask, out])
    return out.numpy().astype(np.float64)


@pytest.mark.parametrize('n,s', _CASES)
def test_stored_metric_has_positive_accurate_physical_spectrum(n, s):
    row = rep._sobolev_circulant_row(n, s, 1e-3)
    # Transform the actual stored coefficients at float64 precision, not an
    # imagined unrounded filter. Do not clip negative or tiny eigenvalues.
    spectrum = np.fft.rfft(row.astype(np.float64))
    expected = _expected_filter(n, s)
    assert (spectrum.real[1:] > 0.).all(), (n, s, spectrum.real[1:].min())
    np.testing.assert_allclose(spectrum.real[1:], expected[1:], rtol=1e-5, atol=0.)
    assert abs(spectrum[0]) < 1e-12
    assert np.max(np.abs(spectrum.imag)) < 1e-12
    # Difference-form convolution subtracts the exact row sum on the diagonal.
    # Verify its actual nontranslation spectrum rather than silently assuming DC=0.
    effective = spectrum.real[1:] - math.fsum(row.astype(np.float64))
    assert (effective > 0.).all()
    np.testing.assert_allclose(effective, expected[1:], rtol=1e-5, atol=0.)


@pytest.mark.parametrize('n,s,k', [(256, 5. / 3., 128), (512, 1.95, 180),
                                    (1024, 1.95, 342), (1024, 1.95, 512)])
@pytest.mark.parametrize('dev', DEVS)
def test_actual_preconditioned_high_mode_remains_positive_and_accurate(n, s, k, dev):
    row = rep._sobolev_circulant_row(n, s, 1e-3)
    if k == n // 2:
        # This input is exactly representable, with no low-mode noise from sin/cos.
        x = np.stack([1. - 2. * (np.arange(n) % 2), np.zeros(n)], axis=1).astype(np.float32)
    else:
        angle = 2 * np.pi * k * np.arange(n) / n
        x = np.stack([np.cos(angle), np.sin(angle)], axis=1).astype(np.float32)
    # Account for quantization of the actual vec2f input. Comparing only with
    # lambda_k*x would mistake its tiny low-frequency rounding for a solver error.
    expected = np.fft.irfft(np.fft.rfft(x.astype(np.float64), axis=0)
                            * _expected_filter(n, s)[:, None], n=n, axis=0)
    actual = _apply(row, x, dev)
    q = float(np.sum(x.astype(np.float64) * actual))
    expected_q = float(np.sum(x.astype(np.float64) * expected))
    assert q > 0., (n, s, k, q, expected_q)
    assert q == pytest.approx(expected_q, rel=1e-4, abs=0.)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-4


@pytest.mark.parametrize('dev', DEVS)
def test_actual_metric_removes_translation_and_preserves_frozen_environment(dev):
    n = 256
    row = rep._sobolev_circulant_row(n, 5. / 3., 1e-3)
    points = np.tile([1.25, -3.5], (2 * n, 1)).astype(np.float32)
    initial = np.tile([7., 11.], (2 * n, 1)).astype(np.float32)
    actual = _apply(row, points, dev, frozen=[0, 1], initial=initial)
    np.testing.assert_array_equal(actual[:n], np.zeros((n, 2)))
    np.testing.assert_array_equal(actual[n:], initial[n:])


def test_metric_that_cannot_resolve_its_own_spectrum_is_rejected_at_construction():
    # No N**2 launch: this only builds/checks a cold FFT row. Even float64 loses
    # positive high modes here, so accepting it would violate the descent metric.
    with pytest.raises(ValueError, match='precision|resolv|positive|condition|represent'):
        rep._sobolev_circulant_row(65536, 1.95, 1e-3)
