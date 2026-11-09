'''
First order RC equivalent digital low pass filter.

The continuous prototype is::

    H_RC(s) = wc / (s + wc),    wc = 2 pi fc_RC

and it is discretised exactly under a zero order hold input assumption, so the
step response matches the continuous one at the sampling instants::

    alpha = exp(-2 pi fc_RC / fs)
    beta  = 1 - alpha = -expm1(-2 pi fc_RC / fs)
    y[k]  = alpha y[k-1] + beta x[k]

``fc_RC`` is the cutoff of the continuous prototype.  The digital -3 dB
frequency of this recursion is only approximately ``fc_RC`` (it converges as
``fc_RC / fs`` tends to zero), so no exact -3 dB claim is made for this filter.
'''
import math

import numpy as np

from .common import (BaselineFilter, ZERO_STATE, check_sample,
                     prepare_frequencies)


class RCFilter(BaselineFilter):
    '''
    Causal first order IIR low pass, the exact ZOH discretisation of an RC.

    Parameters
    ----------
    fs : float
        Sampling frequency, in Hz.  Defaults to 128000.
    fc : float, optional
        Continuous prototype cutoff, in Hz, strictly inside ``(0, fs/2)``.
        When omitted the documented default ``fs/128`` is used and a
        ``UserWarning`` is emitted.
    init_mode : str
        ``'zero_state'`` (default) or ``'steady_state'``.
    initial_value : float, optional
        Required by, and only allowed with, ``'steady_state'``.

    Examples
    --------
    >>> filt = RCFilter(fs=128000.0, fc=1000.0)
    >>> filt.process_sample(0.0)
    0.0
    '''

    cutoff_name = 'fc_RC'

    def _compute_coefficients(self):
        argument = -2.0 * math.pi * self._fc / self._fs
        self._alpha = math.exp(argument)
        # -expm1(x) keeps the relative accuracy of 1 - exp(x) when the
        # argument is small, that is when fc / fs is small.
        self._beta = -math.expm1(argument)

    @property
    def alpha(self):
        '''Feedback coefficient of ``y[k] = alpha y[k-1] + beta x[k]``.'''
        return self._alpha

    @property
    def beta(self):
        '''Feedforward coefficient of ``y[k] = alpha y[k-1] + beta x[k]``.'''
        return self._beta

    @property
    def coefficients(self):
        '''
        Copy of the recursion coefficients.

        The recursion is ``y[k] = alpha y[k-1] + beta x[k]``.
        '''
        return {'alpha': self._alpha, 'beta': self._beta}

    def _reset_state(self, mode, value):
        self._y = 0.0 if mode == ZERO_STATE else value

    def process_sample(self, a_raw):
        '''
        Filter one sample of the converted acceleration sequence.

        ``a_raw`` is the current input in m/s^2; the return value is ``a_RC``
        in m/s^2.  The call advances the internal state.
        '''
        x = check_sample(a_raw)
        y = self._alpha * self._y + self._beta * x
        self._y = y
        return y

    def dc_gain(self):
        '''Gain at DC of the implemented recursion, ``alpha + beta``.'''
        return self._alpha + self._beta

    def frequency_response(self, frequencies):
        '''
        Digital frequency response ``beta / (1 - alpha exp(-j w))``.

        ``w = 2 pi f / fs`` and ``f`` must lie in ``[0, fs/2]``.  A scalar
        frequency returns a scalar complex number, an array returns an array.
        '''
        values, is_scalar = prepare_frequencies(frequencies, self._fs)
        omega = 2.0 * np.pi * values / self._fs
        response = self._beta / (1.0 - self._alpha * np.exp(-1j * omega))
        return response[0] if is_scalar else response
