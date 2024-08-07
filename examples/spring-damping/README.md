# Spring-damping system simulation experiment

Run a spring-damping system simulation with a given set of parameters.

This experiment gives the unit impulse response of a spring-damping system, and try fitting the result to an underdamped ( $0<\zeta<1$ ) second-order system.

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

## File structure

```tree
examples/spring-damping
├── spring-damping-compare.py
├── spring-damping-experiment.py
├── spring-damping-fitting.py
└── spring-damping-simulation.py
```

## Run a full experiment

`spring-damping-experiment.py`

### Usage

```bash
# in project root directory
python examples/spring-damping/spring-damping-experiment.py
```

## Run a simulation without fitting

`spring-damping-simulation.py`

### Usage

Run simulation and save the result:

```bash
# in project root directory
python examples/spring-damping/spring-damping-simulation.py --save
```

## Fitting a result

`spring-damping-fitting.py`

### Usage

```bash
# in project root directory
python examples/spring-damping/spring-damping-fitting.py --data path/to/your/data/
```

## Compare several results in one graph

`spring-damping-compare.py`
