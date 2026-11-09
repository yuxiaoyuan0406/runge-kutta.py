# Closed loop MEMS accelerometer Kalman filter

A discrete time varying Kalman estimator for the closed loop capacitive MEMS
accelerometer of this repository.  The estimator takes the current PID output
`u[k]` as its only online measurement and reconstructs the external
acceleration `a[k]`, together with the displacement `x[k]` and the velocity
`v[k]`.

Two acceleration priors are available:

| `prior_model` | state | acceleration prior | task sheet |
|---|---|---|---|
| `ZOH` | `[x, v, a, I, h]` | constant inside `[t_k, t_{k+1})`, random walk `q_a` | V1.1 (phase 1) |
| `FOH` | `[x, v, a, j, I, h]` | affine inside the interval, Jerk random walk `q_j` | V2.1 (phase 2) |

The implementation follows the frozen equations of the two development task
sheets (`MEMS闭环加速度计_Kalman滤波器开发任务书.docx` V1.1 and
`闭环MEMS加速度计_Kalman滤波器第二阶段FOH开发任务书_公式推导版.docx` V2.1).
It does not modify the existing MEMS–PID simulation, and it never feeds an
estimate back into the controller.

## Status

| Item | State |
|---|---|
| V1.1 sections 12–19, five state ZOH model and recursion | implemented |
| V2.1 sections 11–18, six state FOH model and recursion | implemented |
| Algebraic verification of both paths, section 19.1 | implemented, passing |
| Phase 1 compatibility regression (four records) | implemented, 0.0 deviation |
| Unit tests | 128 tests, passing |
| Data adapter for the existing `data/` layout | implemented |
| Evaluation, PID baseline, `q_a` sweep | implemented |
| FOH / ZOH / baseline comparison (`compare_priors.py`) | implemented |
| Static, step and band-limited dynamic records available | 4 records (one step, three sine) |
| Band-limited **random** acceleration record | **not available** in the workspace, not run |
| Device performance report | see `reports/` of the workspace; conclusions are limited to the four records |

`configs/algebraic-check.json` and `configs/algebraic-check-foh.json` contain
**synthetic** coefficients used only to exercise the code.  They are not device
parameters and no result computed from them is a device result.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Section 9 parameters of both sheets, unit handling, prior model selection, completeness validation |
| `model.py` | ZOH/FOH discretisation, Van Loan covariance, state matrices, observability |
| `kalman.py` | Recursion (14-1)–(14-9) / section 18 plus the review formulas |
| `data_adapter.py` | Loading, unit conversion, alignment, consistency checks |
| `evaluation.py` | Error metrics, PID baseline, NIS, Jerk diagnostics, PSD/ASD, figures |
| `verification.py` | The algebraic checks of section 19.1 of both sheets |
| `regression.py` | The phase 1 ZOH reference record and its comparison |
| `compare_priors.py` | ZOH against FOH against the PID baseline on one record |
| `main.py` | Command line entry point |
| `tests/` | `unittest` suite, including `tests/fixtures/zoh-reference.npz` |
| `configs/parameter-template.json` | Section 9 template for the ZOH path, every value `null` |
| `configs/parameter-template-foh.json` | The same for the FOH path, without `q_a` |
| `configs/xiong-2018*.json` | Device parameters and evaluation windows of the four records |
| `configs/xiong-2018*-foh.json` | The FOH counterpart of each of those files |
| `configs/algebraic-check*.json` | Synthetic coefficients for the self checks |
| `MATH.md` | Symbol dictionary, derivations and numerical evidence of both phases |

The estimator core depends only on NumPy and SciPy.  `data_adapter`,
`evaluation`, `regression` and `compare_priors` are the only modules that touch
files, and only the last two touch matplotlib, so the recursion can be tested
without any simulation code.

## Requirements

```
numpy  scipy  matplotlib
```

`requirements.txt` of the repository already provides them; `simpy` and `tqdm`
are needed by the simulation examples, not by this package.

## Usage

Run from the project root.

```bash
# 0. Which parameters are still missing?  The answer depends on prior_model.
python examples/kalman_accelerometer/main.py params \
    --config examples/kalman_accelerometer/configs/parameter-template-foh.json

# 1. Algebraic verification, ZOH and FOH (no data needed)
python examples/kalman_accelerometer/main.py check \
    --config examples/kalman_accelerometer/configs/xiong-2018.json
python examples/kalman_accelerometer/main.py check \
    --config examples/kalman_accelerometer/configs/xiong-2018-foh.json

# 2. Phase 1 compatibility regression of the ZOH path
python examples/kalman_accelerometer/main.py reference

# 3. Filter a measurement sequence and write the report
python examples/kalman_accelerometer/main.py run \
    --config examples/kalman_accelerometer/configs/xiong-2018-foh.json \
    --data data/20261026-unit_step_1

# 4. Repeat a ZOH run for every entry of q_a_grid (ZOH path only)
python examples/kalman_accelerometer/main.py scan \
    --config examples/kalman_accelerometer/configs/xiong-2018.json \
    --data data/20261026-unit_step_1

# 5. ZOH, FOH and the PID baseline on the same samples and windows
python examples/kalman_accelerometer/compare_priors.py \
    --zoh-config examples/kalman_accelerometer/configs/xiong-2018.json \
    --foh-config examples/kalman_accelerometer/configs/xiong-2018-foh.json \
    --data data/20261026-unit_step_1 -o data/compare-unit-step
```

`run` writes `report.json`, `report.md`, `estimate.npz` and three figures into
`output_dir`.  On the FOH path `estimate.npz` keeps every phase 1 field with its
name, unit and index and appends `jerk_hat`, `jerk_variance`,
`jerk_accel_correlation`, `posterior_variance` and `prior_variance`.

`compare_priors.py` writes `comparison.json`, `comparison.md` and five figures
(overview, window zoom, error, Jerk, ASD), all on the same samples, units and
evaluation windows.

`scan` sweeps `q_a` and therefore refuses an FOH configuration: `q_j` is a
different physical quantity and section 7 of V2.1 forbids an automatic `q_j`
sweep.  Edit the FOH configuration by hand and repeat `run`/`compare_priors.py`.

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

`plot_report.py` draws the phase 1 comparison figures with the repository
`util` tools (`Signal`, `default_time_plot_fig`, `default_noise_plot_fig`):

```bash
python examples/kalman_accelerometer/plot_report.py \
    --config examples/kalman_accelerometer/configs/xiong-2018.json \
    --data data/20261026-unit_step_1 -o figures/unit_step_1
```

`compare_priors.py` draws the phase 2 figures and needs no `util` import.

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

Both templates mirror section 9; every `null` must be replaced by the user and
`params` lists what is missing for the selected `prior_model`.

### ZOH path

* `psd_convention` — `two_sided_psd`, `one_sided_psd` or `one_sided_asd`.
  `mechanical_noise_unit` says whether the value is a force or an acceleration
  density.  The conversion is in section 12.2 and is implemented once in
  `config._convert_noise`; the force and acceleration forms are never added.
* `init_mode` — `k0` when `X0_hat` is the prior of `[x[0], v[0], a[0], I[0], e[-1]]`,
  `k_minus_1` when only `I[-1]` is known.  In the second case the prior is
  shifted with `T_init` (section 16), which also creates the `I[0]`–`e[-1]`
  cross covariance.
* `P0` must be symmetric positive semi definite; a zero matrix is legal and
  means "no information".

### FOH path

* `q_j` — per sample variance of the Jerk boundary increment, unit
  `(m/s^3)^2`.  It is **not** `q_a`: `q_a` is the acceleration random walk of
  the ZOH path and is not read at all here.  The value is a manual entry, see
  section 7 of V2.1; the program has no automatic `q_j` search.
* `jerk0_hat`, `jerk_init_mode` — the Jerk prior.  `known` requires a zero Jerk
  variance and zero Jerk cross covariances, `unknown` requires a finite positive
  variance with a documented basis.
* `P0_FOH` — the complete 6×6 covariance `[x, v, a, j, I, h]`.  As a legal
  assembly route (section 15.1) `P0` plus `jerk_var0` is accepted instead: the
  five state covariance is embedded with the selection matrix `S` and the Jerk
  variance is added.  The assembly does not invent the physical value; the
  device files use `jerk_var0 = 0` because the initial Jerk is known to be zero.
* `state_scales` — one positive scale per state, six entries.  The Jerk scale
  must be set independently of the acceleration scale (section 17.4); with a
  zero Jerk prior variance the default would otherwise be a meaningless unit
  scale.  The device files use `a_scale / Ts`, the acceleration change of one
  sample produced by a Jerk of one scale.

## Observability and detectability

`check` and `run` report, for the selected model, the scaled rank of
`O = [H; HF; ...; HF^(dim-1)]`, the condition number, the PBH test on every
`|λ| ≥ 1`, and whether the acceleration and (on the FOH path) the Jerk
direction lies in `ker O`.

For the device parameters of `xiong-2018` the FOH model has scaled rank 6,
condition number 1.15e6, smallest singular value 2.26e-7, no undetectable mode,
and both the acceleration and the Jerk are observable.  The condition number is
reported because at 128 kHz the Jerk acts on the displacement only through
`Γ_j ≈ T_s^3/6`, so the sixth direction is weak even when it is present in the
rank.  A configuration without meaningful `state_scales` is reported as
numerically weak instead of being silently "observable".

## Tests

```bash
python -m unittest discover -s examples/kalman_accelerometer/tests -t . -v
```

The suite covers:

* discretisation against an independent numerical integral: `Phi`, `Gamma_u`,
  `Gamma_a`, plus the FOH `Gamma_j` against the `(Ts - tau)` integral and the
  `Ts Gamma_a - int tau exp(A_m tau) B_a dtau` identity, the free particle
  limits, and the proof that the naive kernel would be wrong;
* the Van Loan covariance against a numerical Lyapunov integral;
* the explicit layout of `A`, `G`, `L`, `H`, `Q0` of section 13.1 and 18;
* equivalence of (14-4)–(14-5), (14-7)–(14-9) and (15-1)–(15-3) on random
  legal priors, for five and six states;
* a Monte Carlo test that draws one `n[k]` for both the observation and the
  mechanical state, and checks that the residual covariance converges to
  (14-9) and is uncorrelated with the innovation;
* the index and boundary rules of section 16 and 19.1, including the rule that
  `q_j` raises only the Jerk variance of the step that receives it;
* the FOH degenerate case: known zero Jerk and `q_j = 0` reproduces the ZOH
  model with `q_a = 0` on the common states;
* the `q_a` sensitivity rule of the ZOH path — an increase of `q_a` adds
  exactly `Δq_a` to the acceleration prior variance of that step;
* the singular innovation path of section 18;
* configuration validation for both prior models and the data adapter;
* the phase 1 compatibility regression against `tests/fixtures/zoh-reference.npz`.

## Limitations

* The band-limited **random** acceleration class of section 7 has no record in
  this workspace.  The four available records are one step and three sine
  excitations, so the comparison reports static, step and band-limited
  (deterministic) dynamic results; it does not claim to cover band-limited
  random acceleration.
* `q_j` was chosen by hand, one value per record, from a small manually
  enumerated set; the selection and the full sensitivity are in the phase 2
  report.  The code has no automatic sweep or optimisation.
* On the step record the FOH transient is worse than the ZOH one while the
  settled window is much better; no claim of general superiority is made.  The
  step is not a sample path of a Jerk random walk, so its NIS is meaningless at
  the step instant.
* Only the time varying Kalman gain is implemented: no steady state gain, no
  EKF/UKF, no fixed point arithmetic.
* The estimator assumes the independence conditions listed in section 10.1 of
  V2.1.  Whether the simulation actually satisfies them has to be checked
  before the results are interpreted as an exact Bayesian posterior.
* If the closed loop pair `(F, H)` is not detectable for the actual parameters,
  `run` prints a warning; the estimate can then depend on the initial
  condition.  The `check` output reports the scaled rank, the condition number
  and the unobservable modes.
