import simpy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class SpringDampingSystem:
    def __init__(
        self,
        env: simpy.Environment,
        mass: float,
        spring_coef: float,
        damping_coef: float,
        area: float,
        gap: float,
        v_ref: float,
        initial_state: np.ndarray,
        input = None,
    ):
        self.env = env
        self.m = mass
        self.k = spring_coef
        self.b = damping_coef
        self.area = area
        self.gap = gap
        self.v_ref = v_ref
        self.state = initial_state
        self.input = input
        self.simulation_data = {'time': [], 'position': [], 'velocity': []}
        self.output = int(1)

        e0 = 8.854187817e-12
        
        self.elec_force_coef = 0.5 * e0 * (2 * self.v_ref) ** 2 / 10

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
    
    def elec_force(self, x):
        if self.output == 1:
            # pull up
            distance = self.gap - x
            coef = self.elec_force_coef
        else: # self.output == -1
            # pull down
            distance = self.gap + x
            coef = -self.elec_force_coef
        
        return coef / (distance ** 2)
            

    def update(self, dt):
        '''
        Update system state with Runge-Kutta methods.
        '''
        t = self.env.now
        current_state = self.state
        k1 = self.state_equation(current_state, t)
        k2 = self.state_equation(current_state + k1 * dt/2, t)+dt/2
        k3 = self.state_equation(current_state + k2 * dt/2, t)+dt/2
        k4 = self.state_equation(current_state + k3 * dt, t+dt)

        k = (k1 + 2*k2 + 2*k3 + k4)/6
        self.state = current_state + k * dt

    def run_simulation(self, runtime, dt):
        '''
        Run simulation within `runtime`, with time step of `dt`.
        '''
        with tqdm(total=int(runtime/dt), desc='Running simulation') as pbar:
            while self.env.now < runtime:
                self.simulation_data['time'].append(self.env.now)
                self.simulation_data['position'].append(self.state[0])
                self.simulation_data['velocity'].append(self.state[1])
                self.update(dt)
                pbar.update(1)
                yield self.env.timeout(dt)

def external_force(t):
    return 0.00004*np.sin(2 * np.pi * 5e1 * t)

if __name__ == '__main__':
    env = simpy.Environment(0)
    
    initial_state = np.array([0., 0.])
    spring_system = SpringDampingSystem(
        env=env,
        mass=7.45e-7,
        spring_coef=5.623,
        damping_coef=4.95e-6,
        initial_state=initial_state,
        input=external_force
    )

    runtime = 1.
    dt = 1e-6
    env.process(spring_system.run_simulation(runtime, dt))
    env.run(until=runtime)

    plt.figure()
    plt.plot(spring_system.simulation_data['time'], spring_system.simulation_data['position'])
    plt.show()
