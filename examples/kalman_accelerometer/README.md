# Closed loop MEMS accelerometer Kalman filter

A discrete time varying Kalman estimator for the closed loop capacitive MEMS
accelerometer of this repository.  The estimator takes the current PID output
`u[k]` as its only online measurement and reconstructs the external
acceleration `a[k]`, together with the displacement `x[k]` and the velocity
`v[k]`.

The implementation follows the frozen equations of the development task sheet
(`MEMS闭环加速度计_Kalman滤波器开发任务书.docx`, V1.1).  It does not modify the
existing MEMS–PID simulation, and it never feeds an estimate back into the
controller.

## Status

| Item | State |
|---|---|
| Section 12–19 model and recursion | implemented |
| Algebraic verification (section 19.1) | implemented, passing |
| Unit tests | 61 tests, passing |
| Data adapter for the existing `data/` layout | implemented |
| Evaluation, PID baseline, `q_a` sweep | implemented |
| Real device parameters | **not filled in** (section 9 of the task sheet) |
| Device performance report | **not produced** — it needs those parameters and real data |

`configs/algebraic-check.json` contains **synthetic** coefficients used only to
exercise the code.  They are not device parameters and no result computed from
them is a device result.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Section 9 parameters, unit handling, completeness validation |
| `model.py` | Discretisation, Van Loan covariance, five state matrices, observability |
| `kalman.py` | Recursion (14-1)–(14-9) plus the (14-4)/(15-3) review formulas |
| `data_adapter.py` | Loading, unit conversion, alignment, consistency checks |
| `evaluation.py` | Error metrics, PID baseline, NIS, PSD/ASD, figures |
| `verification.py` | The algebraic checks of section 19.1 |
| `main.py` | Command line entry point |
| `tests/` | `unittest` suite |
| `configs/parameter-template.json` | Section 9 template with every value left `null` |
| `configs/xiong-2018.json` | Parameters derived from `parameters/xiong-2018.json` |
| `configs/algebraic-check.json` | Synthetic coefficients for the self checks |
| `MATH.md` | Symbol dictionary, derivation and numerical evidence |

The estimator core depends only on NumPy and SciPy.  `data_adapter` and
`evaluation` are the only modules that touch files or matplotlib, so the
recursion can be tested without any simulation code.

## Requirements

```
numpy  scipy  matplotlib
```

`requirements.txt` of the repository already provides them; `simpy` and `tqdm`
are needed by the simulation examples, not by this package.

## Usage

Run from the project root.

```bash
# 1. What does the parameter file still need?
python examples/kalman_accelerometer/main.py params \
    --config examples/kalman_accelerometer/configs/parameter-template.json

# 2. Algebraic verification of the implementation (no data needed)
python examples/kalman_accelerometer/main.py check \
    --config examples/kalman_accelerometer/configs/algebraic-check.json

# 3. Filter a measurement sequence and write the report
python examples/kalman_accelerometer/main.py run \
    --config path/to/device.json --data data/20260126-113329-spr-dmp-pid-test

# 4. Repeat the run for every entry of q_a_grid
python examples/kalman_accelerometer/main.py scan \
    --config path/to/device.json --data data/20260126-113329-spr-dmp-pid-test
```

`run` writes `report.json`, `report.md`, `estimate.npz` and three figures into
`output_dir`.

### Converting MATLAB data

`mat_to_npy.py` turns a `.mat` file into a tree of `.npy` arrays:

```bash
python examples/kalman_accelerometer/mat_to_npy.py \
    matlab_data/unit_step_1/out.mat -o data/unit_step_1
```

A plain MAT file (arrays, structs, char, cell) is read with `scipy` and written
directly.  A Simulink `SimulationOutput`, which is what `out = sim(model)`
produces, is an MCOS object: `scipy` can only recognise it, not decode it.  For
those the script calls MATLAB in batch mode once to flatten the object and then
converts the result.  When MATLAB cannot be found the script writes
`mat_to_npy_export.m` and prints the exact command to run.

```bash
# skip the high rate solver signals
python examples/kalman_accelerometer/mat_to_npy.py out.mat -o out \
    --exclude tout,a_in,chopout8,displacement,pid
```

The output directory holds `raw/<signal>.npy` for every signal plus a
`manifest.json` with shape, dtype, sample count and sampling period.  With
`--layout repo` (the default) the two signals used by this example are also
written in the simulation layout:

| File | Signal |
|---|---|
| `pid/time.npy`, `pid/pid.npy` | `mems_out` |
| `mass_block/time.npy`, `mass_block/acceleration.npy` | `a_true` |

so `column_map` can point `a_true` at `mass_block/acceleration.npy`.

### Figures

`plot_report.py` draws the comparison figures with the repository `util`
tools (`Signal`, `default_time_plot_fig`, `default_noise_plot_fig`):

```bash
python examples/kalman_accelerometer/plot_report.py \
    --config examples/kalman_accelerometer/configs/xiong-2018.json \
    --data data/20261026-unit_step_1 -o figures/unit_step_1
```

It writes an overview, a two panel zoom of the static and settled windows, the
error comparison and the amplitude spectral density of the two error signals.
`util.compute_asd_welch` is avoided on purpose; the ASD is computed with
`scipy.signal.welch` inside the script.


## Data layout

The adapter reads the layout written by the simulation examples, that is a run
directory with per-group `.npy` files:

```
data/<run>/param.json
data/<run>/pid/time.npy
data/<run>/pid/pid.npy
data/<run>/mass_block/position.npy
data/<run>/mass_block/velocity.npy
```

`column_map` in the configuration renames these.  `a_true` is optional: the
simulation does not store the external acceleration, so a true acceleration
column has to be provided by the user (or generated by the experiment script)
before an RMSE against truth can be reported.  `units` applies a multiplicative
factor per column, `sample_alignment` trims and shifts the record.

A mismatch between the data sampling period and `fs` is an error, not a silent
rescale.

## Filling in the parameters

`configs/parameter-template.json` mirrors section 9.  Every `null` must be
replaced by the user; `params` lists what is missing.  Two entries deserve
attention:

* `psd_convention` — `two_sided_psd`, `one_sided_psd` or `one_sided_asd`.
  `mechanical_noise_unit` says whether the value is a force or an acceleration
  density.  The conversion is in section 12.2 and is implemented once in
  `config._convert_noise`; the force and acceleration forms are never added.
* `init_mode` — `k0` when `X0_hat` is the prior of `[x[0], v[0], a[0], I[0], e[-1]]`,
  `k_minus_1` when only `I[-1]` is known.  In the second case the prior is
  shifted with `T_init` (section 16), which also creates the `I[0]`–`e[-1]`
  cross covariance.

`P0` must be symmetric positive semi definite; a zero matrix is legal and means
"no information".

## Tests

```bash
python -m unittest discover -s examples/kalman_accelerometer/tests -t . -v
```

The suite covers:

* discretisation against an independent numerical integral, the free particle
  limit (19-1), the Van Loan covariance against a numerical Lyapunov integral;
* the explicit layout of `A`, `G`, `L`, `H`, `Q0` of section 13.1;
* equivalence of (14-4)–(14-5), (14-7)–(14-9) and (15-1)–(15-3) on random
  legal priors;
* a Monte Carlo test that draws one `n[k]` for both the observation and the
  mechanical state, and checks that the residual covariance converges to (14-9)
  and is uncorrelated with the innovation;
* the index and boundary rules of section 16 and 19.1;
* the `q_a` sensitivity rule — an increase of `q_a` adds exactly `Δq_a` to the
  acceleration prior variance of that step and nothing else;
* the singular innovation path of section 18;
* configuration validation and the data adapter.

## Limitations

* No real parameters and no real data were supplied, so this package makes no
  claim about noise reduction, bandwidth or convergence time on a device.
* Only the time varying Kalman gain is implemented: no steady state gain, no
  EKF/UKF, no fixed point arithmetic.
* The estimator assumes the independence conditions listed in section 11.
  Whether the simulation actually satisfies them has to be checked before the
  results are interpreted as an exact Bayesian posterior.
* If the closed loop pair `(F, H)` is not detectable for the actual parameters,
  `run` prints a warning; the estimate can then depend on the initial
  condition.  The `check` output reports the scaled rank, the condition number
  and the unobservable modes.
