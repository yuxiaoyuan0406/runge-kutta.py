'''
Closed loop MEMS accelerometer Kalman filter.

The package implements sections 12 to 19 of the development task sheet:

``config``
    Section 9 parameters, units and completeness validation.
``model``
    Exact zero order hold discretisation, Van Loan process covariance, the
    five state matrices and the observability / detectability checks.
``kalman``
    The discrete time varying recursion (14-1)-(14-9) with the review formulas
    kept as independent references.
``data_adapter``
    Loading, unit conversion and alignment of the existing simulation files.
``evaluation``
    Error metrics, PID baseline, innovation statistics and figures.
``verification``
    The algebraic checks required by section 19.1.
'''
from .config import Config, ConfigError, build_config, load_config
from .model import ClosedLoopModel, ModelError
from .kalman import ClosedLoopKalman, FilterError, KalmanStep

__all__ = [
    'Config',
    'ConfigError',
    'build_config',
    'load_config',
    'ClosedLoopModel',
    'ModelError',
    'ClosedLoopKalman',
    'FilterError',
    'KalmanStep',
]
