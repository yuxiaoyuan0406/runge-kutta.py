import simpy
import numpy as np

class SpringDampingSystem:
    def __init__(
        self,
        env: simpy.Environment,
        mass: float,
        spring_coef: float,
        damping_coef: float,
        initial_state: np.ndarray = np.array([0.,0.]),
        runtime: float=1.,
        dt: float=1e-6,
        input_force = None,
    ):
        self.env = env
        self.m = mass
        self.k = spring_coef
        self.b = damping_coef
        self.state = initial_state
        self.runtime = runtime
        self.dt = dt
        self.input = input_force
        self.simulation_data = {'time': [], 'position': [], 'velocity': []}
        self.pid_cmd = int(1)

        self.env.process(self.run(runtime, dt))

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
            f_external = self.input(t)
        else:
            f_external = 0
        a = (f_external - self.k * x - self.b * v) / self.m

        return np.array([v,a])

    def update(self, dt):
        '''
        Update system state with Runge-Kutta methods.
        '''
        t = self.env.now
        current_state = self.state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * dt/2, t+dt/2)
        k3 = self.state_equation(current_state + k2 * dt/2, t+dt/2)
        k4 = self.state_equation(current_state + k3 * dt, t+dt)

        k = (k1 + 2*k2 + 2*k3 + k4)/6
        self.state = current_state + k * dt

    def run(self, runtime, dt):
        '''
        Run simulation within `runtime`, with time step of `dt`.
        '''
        while self.env.now < runtime:
            # self.pid_cmd = yield
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['position'].append(self.state[0])
            self.simulation_data['velocity'].append(self.state[1])
            self.update(dt)
            yield self.env.timeout(dt)