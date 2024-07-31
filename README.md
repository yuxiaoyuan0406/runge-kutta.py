# Runge-Kutta in Python

Run Runge-Kutta methods in python.

## TOC

- [Runge-Kutta in Python](#runge-kutta-in-python)
  - [TOC](#toc)
  - [Experiments and examples](#experiments-and-examples)
    - [Spring-damping system simulation experiment](#spring-damping-system-simulation-experiment)

## Experiments and examples

All examples are placed under `examples/`

### Spring-damping system simulation experiment

Run a spring-damping system simulation with a given set of parameters.

This experiment gives the unit impulse response of a spring-damping system, and try fitting the result to an underdamped ( $0<\zeta<1$ ) second-order system.

See `examples/spring-damping/` for more details.

To run this experiment:

```bash
# in project root directory
python examples/spring-damping/spring-damping-experiment.py
```

Use your own parameters:

```bash
# in project root directory
python examples/spring-damping/spring-damping-experiment.py --param path/to/your/para.json
```
