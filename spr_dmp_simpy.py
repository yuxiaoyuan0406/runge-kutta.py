import simpy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from module import SpringDampingSystem

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
        area=0,
        gap=0,
        v_ref=2.5,
        initial_state=initial_state,
        input=external_force
    )

    runtime = 1.
    dt = 1e-6

    env.process(spring_system.run(runtime, dt))
    with tqdm(total=int(runtime/dt), desc='Running simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)

    plt.figure()
    plt.plot(spring_system.simulation_data['time'], spring_system.simulation_data['position'])
    plt.show()
