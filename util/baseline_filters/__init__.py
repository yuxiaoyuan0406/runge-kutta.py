'''
Baseline low pass filters for the closed loop MEMS accelerometer comparison.

Two mutually independent, causal and reproducible references:

``RCFilter``
    First order RC equivalent IIR, exact ZOH discretisation of ``wc/(s+wc)``.
``ButterworthFilter``
    Second order Butterworth IIR, bilinear transform with prewarping, so the
    configured cutoff is the digital -3 dB frequency.

Both take a sequence ``a_raw[k]`` that is already expressed in m/s^2, keep
their own state between calls and never touch the Kalman estimator::

    from util.baseline_filters import RCFilter, ButterworthFilter

    rc = RCFilter(fs=128000.0, fc=1000.0)
    bw = ButterworthFilter(fs=128000.0, fc=1000.0)
    a_rc = rc.process_array(a_raw)
    a_bw = bw.process_array(a_raw)

Neither filter converts units, removes a trend, repairs outliers, resamples or
filters backwards; those operations would change the interpretability of a
benchmark.  Noise RMS, RMSE and group delay are left to the comparison
platform, not computed here.

Note that importing this subpackage imports the parent ``util`` package, which
in turn imports matplotlib through ``util.plot``.
'''
from .butterworth import ButterworthFilter
from .common import (DEFAULT_FC_DIVISOR, DEFAULT_FS, INIT_MODES, STEADY_STATE,
                     ZERO_STATE)
from .rc import RCFilter

__all__ = [
    'ButterworthFilter',
    'RCFilter',
    'DEFAULT_FS',
    'DEFAULT_FC_DIVISOR',
    'INIT_MODES',
    'ZERO_STATE',
    'STEADY_STATE',
]
