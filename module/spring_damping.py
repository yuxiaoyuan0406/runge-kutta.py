'''
Spring damping system.
'''
import inspect
from typing import Callable, Optional, Any

import os
import simpy
import numpy as np
from .system import SystemState
from .base import ModuleBase
from .noise import Noise

USING_CPP_BACKEND = False

class SpringDampingSystem(ModuleBase):
    '''
    Spring damping system.
    '''

    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        mass: float,
        spring_coef: float,
        damping_coef: float,
        initial_state: np.ndarray = np.array([0., 0.], dtype=np.float64),
        runtime: float = 1.,
        dt: float = 1e-6,
        input_accel=None,
        thermal_noise=None,
        output_noise=None,
    ):
        super().__init__(env=env, runtime=runtime, dt=dt)
        self.m = mass
        self.k = spring_coef
        self.b = damping_coef
        self.system_state = system_state
        self.state = np.array(initial_state)
        self._input_raw = input_accel
        self.input = self._wrap_input(input_accel)
        self.simulation_data = {'time': [], 'position': [], 'velocity': []}
        self.pid_cmd = int(1)

        if thermal_noise is None:
            thermal_noise = Noise(noise_power=0, sample_time=dt, mean=0)
        self.thermal_noise = thermal_noise
        if output_noise is None:
            output_noise = Noise(noise_power=0, sample_time=dt, mean=0)
        self.output_noise = output_noise

        # backend handler
        global USING_CPP_BACKEND
        if USING_CPP_BACKEND:
            try: 
                from .backend import SpringDamping
                self.__backend = SpringDamping.SpringDampingBackend(self.m, self.k, self.b)
                self.__state_equation = self.__backend.state_equation
                
                self.__ode4 = SpringDamping.ode4
            except Exception:
                USING_CPP_BACKEND = False

    @classmethod
    def init_by_dict(
        cls,
        env: simpy.Environment,
        system_state: SystemState,
        param: dict[str, float],
        input_accel=None,
        thermal_noise=None,
        output_noise=None,
        ) -> "SpringDampingSystem" :
        """Initialize a object by a `dict`.

        Args:
            env (simpy.Environment): Simulation env.
            system_state (SystemState): Top module system state.
            param (dict[str, float]): Parameter definations.
            input_accel (_type_, optional): Defaults to None.
            thermal_noise (_type_, optional): Defaults to None.
            output_noise (_type_, optional): Defaults to None.

        Returns:
            SpringDampingSystem: The object.
        """
        runtime = param.get('runtime', 1.)
        dt = param.get('mechanic_dt', 1e-6)
        mass = param['mass']
        spring_coef = param['spring_coef']
        damping_coef = param['damping_coef']
        initial_state = np.array(param.get('initial_state', [0, 0]), dtype=np.float64)
        ret = SpringDampingSystem(
            env=env, 
            system_state=system_state, 
            mass=mass,
            spring_coef=spring_coef,
            damping_coef=damping_coef,
            initial_state=initial_state,
            runtime=runtime,
            dt=dt,
            input_accel=input_accel,
            thermal_noise=thermal_noise,
            output_noise=output_noise)

        return ret

    @staticmethod
    def _wrap_input(fn: Optional[Callable[..., float]]):
        if fn is None:
            return None

        sig = inspect.signature(fn)

        # 1) 优先尝试 (t, x) 两个位置参数（最通用：不依赖参数名）
        try:
            sig.bind(0.0, 0.0)
            mode = "pos2"
        except TypeError:
            # 2) 再尝试仅 (t)
            try:
                sig.bind(0.0)
                mode = "pos1"
            except TypeError:
                # 3) 再尝试关键字（适配 def f(t, *, x): ... 或 def f(*, t, x): ...）
                try:
                    sig.bind(t=0.0, x=0.0)
                    mode = "kw_tx"
                except TypeError:
                    # 兜底：让错误尽早暴露
                    raise TypeError(
                        "input_accel must be callable like f(t) or f(t, x) (or accept keywords t,x)."
                    )

        if mode == "pos2":
            def _call(t: float, x: float, state: Any):
                return fn(t, x)
            return _call

        if mode == "kw_tx":
            def _call(t: float, x: float, state: Any):
                return fn(t=t, x=x)
            return _call

        # mode == "pos1"
        def _call(t: float, x: float, state: Any):
            return fn(t)
        return _call

    def __str__(self):
        return f'SpringDampingSystem(m={self.m}, k={self.k}, b={self.b})'
    def __repr__(self):
        return self.__str__()

    def __state_equation(self, state, a_ext):
        x, v = state
        a = a_ext - (self.k * x + self.b * v) / self.m

        return np.array([v, a])

    def __ode4(self, _k):
        return (_k[0] + 2 * _k[1] + 2 * _k[2] + _k[3]) / 6

    def state_equation(self, state, t):
        '''
        The state space equation of the system.
        ```
        dy/dt = f(y,t)
                ^      
        ```
        '''
        if self.input:
            a_external = self.input(t, state[0], state)
        else:
            a_external = 0
        return self.__state_equation(state, a_external)

    def predict_state(self, dt):
        '''
        Predict the system state after a time step `dt` using the current state.

        Args:
            dt (float): The time step for prediction.

        Returns:
            np.ndarray: The predicted state [position, velocity] after `dt`.
        '''
        t = self.env.now
        current_state = self.state
        _k = np.empty((4, 2), dtype=np.float64)
        _k[0] = self.state_equation(current_state, t)
        _k[1] = self.state_equation(current_state + _k[0] * dt / 2, t + dt / 2)
        _k[2] = self.state_equation(current_state + _k[1] * dt / 2, t + dt / 2)
        _k[3] = self.state_equation(current_state + _k[2] * dt, t + dt)

        k = self.__ode4(_k)
        k += self.thermal_noise.next() * np.array([0, 1], dtype=np.float64)  # Add noise to the acceleration
        predicted_state = current_state + k * dt
        return predicted_state

    def update(self):
        '''
        Update system state with Runge-Kutta methods.
        '''
        self.state = self.predict_state(self.dt)

    def run(self):
        '''
        Execute the simulation for the specified `runtime`, advancing in steps of `dt`.
        '''
        while self.env.now < self.runtime:
            # self.pid_cmd = yield
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['position'].append(self.state[0] + self.output_noise.next() * self.m / self.k)
            self.simulation_data['velocity'].append(self.state[1])
            self.system_state.mass_block_state = self.state
            self.update()
            yield self.env.timeout(self.dt)

    def save(self, directory):
        """Save simulation results.

        Args:
            dir (str): Directory to save.
        """
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))
        np.save(os.path.join(directory, 'position'),
                np.array(self.simulation_data['position']))
        np.save(os.path.join(directory, 'velocity'),
                np.array(self.simulation_data['velocity']))

    @classmethod
    def load_from_file(cls, directory):
        """加载模拟数据。

        Args:
            directory (str): 数据存储的目录。

        Returns:
            tuple[util.Signal]: 包含位置和速度信号的元组。
        """
        t = np.load(os.path.join(directory, 'time.npy'))
        posi = np.load(os.path.join(directory, 'position.npy'))
        velo = np.load(os.path.join(directory, 'velocity.npy'))
        # import sys
        # sys.path.append(os.path.abspath('../..'))
        from util import Signal
        posi = Signal(posi, t=t, label='Displacement')
        velo = Signal(velo, t=t, label='Velocity')
        return posi, velo
