# Baseline low pass filters

Two independent, causal digital low pass filters used as references for the
closed loop MEMS accelerometer Kalman estimator of
`examples/kalman_accelerometer`:

| Filter | Continuous prototype | Discretisation |
|---|---|---|
| `RCFilter` | `H(s) = wc / (s + wc)` | exact ZOH step response matching |
| `ButterworthFilter` | `H(s) = wc^2 / (s^2 + sqrt(2) wc s + wc^2)` | bilinear transform with prewarping |

They are pure numerical filters. They do not import the Kalman module, they
never change the estimator, and they do not convert units: the input
`a_raw[k]` is already an acceleration in m/s².

## Interface

Both classes share the same API.

```python
from util.baseline_filters import ButterworthFilter, RCFilter

filt = RCFilter(fs=128000.0, fc=1000.0)          # or ButterworthFilter(...)
a_rc = filt.process_array([0.0, 1.0, 0.0])       # whole sequence
y = filt.process_sample(0.5)                     # one sample, state advances
filt.reset()                                     # clear history, keep fs/fc
filt.reconfigure(fc=2000.0)                      # new coefficients, state reset
filt.coefficients                                # dict of the recursion
filt.frequency_response(1000.0)                  # H(e^jw), w = 2 pi f / fs
```

| Member | Meaning |
|---|---|
| `fs` | sampling frequency, Hz |
| `fc` | configured cutoff, Hz |
| `coefficients` | `{'alpha', 'beta'}` for RC, `{'b0', 'b1', 'b2', 'a1', 'a2'}` for Butterworth |
| `process_sample(a_raw)` | one sample in, one sample out, in m/s² |
| `process_array(a_raw)` | sequential loop over `process_sample`, same state rules |
| `reset(init_mode=None, initial_value=None)` | clear the state, keep `fs`, `fc` and the coefficients |
| `reconfigure(fs=None, fc=None)` | recompute the coefficients **and** reset the state |
| `frequency_response(f)` | complex `H(e^{j 2 pi f / fs})` for `f` in `[0, fs/2]` |
| `poles` (Butterworth) | both poles of the digital filter |
| `dc_gain()` | gain at DC of the implemented recursion |

`process_array` is a loop, not a vectorised filter: it must stay bit identical
to repeated `process_sample` calls and it must stay causal.

## Configuration and defaults

* `fs` defaults to **128000 Hz**.
* `fc` defaults to **`fs/128`** (1000 Hz at 128 kHz) but it is never silent:
  omitting `fc` emits a `UserWarning` telling which default was used. Pass an
  explicit value for any real comparison.
* `fc` must satisfy `0 < fc < fs/2`; the Nyquist frequency itself is refused.
* `reconfigure` refuses a parameter that would leave `fc` outside `(0, fs/2)`.

## Initialisation

`zero_state` (default) starts from zero history.

`steady_state` requires an explicit `initial_value = c` and starts from the
equilibrium of the constant input `c`, so the first output is already `c`:

* RC: `y[-1] = c`.
* Butterworth: `s1[-1] = (1 - b0) c`, `s2[-1] = (b2 - a2) c`.

The value is never guessed from a future sample. `zero_state` refuses an
`initial_value` instead of ignoring it.

## Equations

### RC

```
alpha = exp(-2 pi fc / fs)
beta  = -expm1(-2 pi fc / fs) = 1 - alpha
y[k]  = alpha y[k-1] + beta x[k]
H(e^jw) = beta / (1 - alpha e^-jw)
```

`fc` is the cutoff of the continuous prototype; the digital -3 dB frequency is
only approximately `fc` and converges as `fc/fs -> 0`. `expm1` keeps `beta`
accurate when `fc << fs`.

### Butterworth

```
K  = tan(pi fc / fs)
D  = 1 + sqrt(2) K + K^2
b0 = K^2 / D,  b1 = 2 K^2 / D,  b2 = K^2 / D
a1 = 2 (K^2 - 1) / D,  a2 = (1 - sqrt(2) K + K^2) / D
```

Direct form I reference recursion, with the monic denominator
`1 + a1 z^-1 + a2 z^-2` and negative feedback:

```
y[k] = b0 x[k] + b1 x[k-1] + b2 x[k-2] - a1 y[k-1] - a2 y[k-2]
```

The implementation uses the transposed direct form II:

```
y[k]  = b0 x[k] + s1[k-1]
s1[k] = b1 x[k] - a1 y[k] + s2[k-1]
s2[k] = b2 x[k] - a2 y[k]
```

The two forms are cross checked in the tests. `a1`, `a2` are stored with the
monic convention, not pre-negated.

## Reference numbers

`fs = 128000 Hz`, `fc = 1000 Hz`:

| Coefficient | Value |
|---|---|
| RC `alpha` | 0.9520979267837046 |
| RC `beta` | 0.04790207321629536 |
| BW `b0` | 0.0005820761342360482 |
| BW `b1` | 0.0011641522684720964 |
| BW `b2` | 0.0005820761342360482 |
| BW `a1` | -1.930606427219668 |
| BW `a2` | 0.9329347317566122 |

## Tests

Run from the project root:

```bash
python3 -m unittest discover -s util/tests -t . -p "test_baseline_*.py" -v
```

The suite covers the coefficients above to 1e-12, zero and constant inputs in
both initialisation modes, impulse and step responses against independent
reference recursions, batch versus sample consistency, `reset` consistency,
the Butterworth -3.0103 dB point at `fc` within 0.01 dB, an independent
time-domain sine measurement of both magnitudes, pole stability, the
causality of a prefix, and the parameter validation.

## Assumptions and limitations

* The input is already an acceleration in m/s². Unit conversion, `u[k]` to
  acceleration scaling and the choice of `a_raw` belong to the comparison
  platform, not to this package.
* No detrending, outlier repair, resampling or zero phase filtering.
* No noise RMS, RMSE, group delay computation and no automatic cutoff search;
  those are comparison metrics and are intentionally out of scope.
* A single cutoff is configured per instance. A comparison that needs several
  cutoffs creates several instances.
* The Butterworth coefficients are refused when the pole at `z = 1` is not
  representable in double precision (`fc/fs` below roughly 1e-17); the message
  names the constraint instead of returning a silently unstable filter.
* Importing `util.baseline_filters` imports the parent `util` package and
  therefore matplotlib through `util.plot`.
