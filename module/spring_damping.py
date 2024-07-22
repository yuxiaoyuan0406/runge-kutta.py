'''
Spring damping system.
'''
import simpy
import numpy as np
from .system import SystemState


class SpringDampingSystem:
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
    ):
        self.env = env
        self.m = mass
        self.k = spring_coef
        self.b = damping_coef
        self.system_state = system_state
        self.state = initial_state
        self.runtime = runtime
        self.dt = dt
        self.input = input_accel
        self.simulation_data = {'time': [], 'position': [], 'velocity': []}
        self.pid_cmd = int(1)

        self.env.process(self.run(self.dt))

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
        a = a_external - (self.k * x - self.b * v) / self.m

        return np.array([v, a])

    def update(self, dt):
        '''
        Update system state with Runge-Kutta methods.
        '''
        t = self.env.now
        current_state = self.state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * dt / 2, t + dt / 2)
        k3 = self.state_equation(current_state + k2 * dt / 2, t + dt / 2)
        k4 = self.state_equation(current_state + k3 * dt, t + dt)

        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state = current_state + k * dt

    def run(self, dt):
        '''
        Run simulation within `runtime`, with time step of `dt`.
        '''
        while self.env.now < self.runtime:
            # self.pid_cmd = yield
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['position'].append(self.state[0])
            self.simulation_data['velocity'].append(self.state[1])
            self.system_state.mass_block_state = self.state
            self.update(dt)
            yield self.env.timeout(dt)
