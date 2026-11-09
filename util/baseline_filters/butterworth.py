'''
Second order Butterworth digital low pass filter.

The continuous prototype is::

    H_BW(s) = wc^2 / (s^2 + sqrt(2) wc s + wc^2)

and it is discretised with the bilinear transform *with prewarping*, so that
the configured ``fc_BW`` is the digital -3 dB cutoff (up to floating point
accuracy)::

    K  = tan(pi fc_BW / fs)
    D  = 1 + sqrt(2) K + K^2
    b0 = K^2 / D,   b1 = 2 K^2 / D,   b2 = K^2 / D
    a1 = 2 (K^2 - 1) / D,   a2 = (1 - sqrt(2) K + K^2) / D

    H_BW(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
    y[k] = b0 x[k] + b1 x[k-1] + b2 x[k-2] - a1 y[k-1] - a2 y[k-2]

The denominator is monic and the feedback terms carry a minus sign; the stored
``a1``, ``a2`` are *not* pre-negated as some filter libraries do.

The state update is the transposed direct form II::

    y[k]  = b0 x[k] + s1[k-1]
    s1[k] = b1 x[k] - a1 y[k] + s2[k-1]
    s2[k] = b2 x[k] - a2 y[k]
'''
import math

import numpy as np

from .common import (BaselineFilter, STEADY_STATE, ZERO_STATE, check_sample,
                     prepare_frequencies)

_SQRT2 = math.sqrt(2.0)


class ButterworthFilter(BaselineFilter):
    '''
    Causal second order Butterworth IIR low pass, transposed direct form II.

    Parameters
    ----------
    fs : float
        Sampling frequency, in Hz.  Defaults to 128000.
    fc : float, optional
        Digital -3 dB cutoff, in Hz, strictly inside ``(0, fs/2)``.  When
        omitted the documented default ``fs/128`` is used and a ``UserWarning``
        is emitted.
    init_mode : str
        ``'zero_state'`` (default) or ``'steady_state'``.
    initial_value : float, optional
        Required by, and only allowed with, ``'steady_state'``.

    Examples
    --------
    >>> filt = ButterworthFilter(fs=128000.0, fc=1000.0)
    >>> filt.process_sample(0.0)
    0.0
    '''

    cutoff_name = 'fc_BW'

    def _compute_coefficients(self):
        k = math.tan(math.pi * self._fc / self._fs)
        k2 = k * k
        denominator = 1.0 + _SQRT2 * k + k2
        b0 = k2 / denominator
        b1 = 2.0 * k2 / denominator
        b2 = k2 / denominator
        a1 = 2.0 * (k2 - 1.0) / denominator
        a2 = (1.0 - _SQRT2 * k + k2) / denominator

        poles = np.roots([1.0, a1, a2])
        if np.max(np.abs(poles)) >= 1.0:
            raise ValueError(
                f'fc_BW = {self._fc} Hz at fs = {self._fs} Hz is too far below '
                'the sampling rate for a double precision biquad: the pole at '
                'z = 1 cannot be represented, so the filter would not be '
                'stable.  Use a larger fc/fs ratio.')
        if 1.0 + a1 + a2 <= 0.0:
            raise ValueError(
                f'fc_BW = {self._fc} Hz at fs = {self._fs} Hz has a '
                'non positive denominator at DC; the coefficients are not '
                'representable in double precision.')

        self._b0, self._b1, self._b2 = b0, b1, b2
        self._a1, self._a2 = a1, a2

    @property
    def coefficients(self):
        '''
        Copy of the biquad coefficients.

        ``H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)``; the
        feedback terms of the recursion use ``-a1 y[k-1] - a2 y[k-2]``.
        '''
        return {
            'b0': self._b0,
            'b1': self._b1,
            'b2': self._b2,
            'a1': self._a1,
            'a2': self._a2,
        }

    @property
    def b(self):
        '''Feedforward coefficients ``(b0, b1, b2)``.'''
        return (self._b0, self._b1, self._b2)

    @property
    def a(self):
        '''Feedback coefficients ``(a1, a2)`` with the monic convention.'''
        return (self._a1, self._a2)

    @property
    def poles(self):
        '''Both poles of the digital filter.'''
        return np.roots([1.0, self._a1, self._a2])

    def dc_gain(self):
        '''Gain at DC, ``(b0 + b1 + b2) / (1 + a1 + a2)``.'''
        return ((self._b0 + self._b1 + self._b2) /
                (1.0 + self._a1 + self._a2))

    def _reset_state(self, mode, value):
        if mode == ZERO_STATE or value is None:
            self._s1 = 0.0
            self._s2 = 0.0
        else:
            # Constant input c has the steady state y = c, so
            # s1 = (1 - b0) c and s2 = (b2 - a2) c give y[0] = c exactly
            # because the DC gain is one.
            self._s1 = (1.0 - self._b0) * value
            self._s2 = (self._b2 - self._a2) * value

    def process_sample(self, a_raw):
        '''
        Filter one sample of the converted acceleration sequence.

        ``a_raw`` is the current input in m/s^2; the return value is ``a_BW``
        in m/s^2.  The call advances the internal state.
        '''
        x = check_sample(a_raw)
        y = self._b0 * x + self._s1
        self._s1 = self._b1 * x - self._a1 * y + self._s2
        self._s2 = self._b2 * x - self._a2 * y
        return y

    def frequency_response(self, frequencies):
        '''
        Digital frequency response of the biquad.

        ``z^-1 = exp(-j w)`` with ``w = 2 pi f / fs`` and ``f`` in
        ``[0, fs/2]``.  A scalar frequency returns a scalar complex number, an
        array returns an array.
        '''
        values, is_scalar = prepare_frequencies(frequencies, self._fs)
        omega = 2.0 * np.pi * values / self._fs
        z_inv = np.exp(-1j * omega)
        numerator = self._b0 + self._b1 * z_inv + self._b2 * z_inv * z_inv
        denominator = 1.0 + self._a1 * z_inv + self._a2 * z_inv * z_inv
        response = numerator / denominator
        return response[0] if is_scalar else response
