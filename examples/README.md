# Examples

Code usage examples.

| Directory | Content |
|---|---|
| `spring-damping/` | Spring-damping simulation, experiments and fitting |
| `noise/` | Noise injection and Kalman filtering of the spring-damping system |
| `pid/` | PID experiments on the spring-damping system |
| `xiong-2018/` | Closed loop PID example following Xiong 2018 |
| `kalman/` | Small alpha-beta-gamma filter demo |
| `kalman_accelerometer/` | Closed loop MEMS accelerometer Kalman filter (task sheet V1.1) |

`kalman_accelerometer/` is a package rather than a single script: the folder
name is an identifier so that the modules can use relative imports.  See its
`README.md` and `MATH.md`.
