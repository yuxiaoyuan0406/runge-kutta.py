'''
Spring damping system.
'''
import os
import simpy
import numpy as np
from .system import SystemState
from .base import ModuleBase
from .noise import Noise


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
        noise=Noise(noise_power=0, sample_time=1e-6, mean=0),
    ):
        super().__init__(env=env, runtime=runtime, dt=dt)
        self.m = mass
        self.k = spring_coef
        self.b = damping_coef
        self.system_state = system_state
        self.state = initial_state
        self.input = input_accel
        self.simulation_data = {'time': [], 'position': [], 'velocity': []}
        self.pid_cmd = int(1)

        self.noise = noise

    def __str__(self):
        return f'SpringDampingSystem(m={self.m}, k={self.k}, b={self.b})'
    def __repr__(self):
        return self.__str__()


    def state_equation(self, state, t):
        '''
        The state space equation of the system.
        ```
        dy/dt = f(y,t)
                ^      
        ```
        '''
        x, v = state
        if self.input:
            a_external = self.input(t)
        else:
            a_external = 0
        a = a_external - (self.k * x + self.b * v) / self.m + self.noise.next()

        return np.array([v, a])

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
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * dt / 2, t + dt / 2)
        k3 = self.state_equation(current_state + k2 * dt / 2, t + dt / 2)
        k4 = self.state_equation(current_state + k3 * dt, t + dt)

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
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
            self.simulation_data['position'].append(self.state[0])
            self.simulation_data['velocity'].append(self.state[1])
            self.system_state.mass_block_state = self.state
            self.update()
            yield self.env.timeout(self.dt)

    def save(self, directory):
        """Save simulation results.

        Args:
            dir (str): Directory to save.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
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
        posi = Signal(posi, t=t)
        velo = Signal(velo, t=t)
        return posi, velo
