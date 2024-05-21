'''
Author: Yu Xiaoyuan
'''
import simpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import module

matplotlib.use('TkAgg')

def external_force(t: float):
    '''
    External Force, nothing unusual.
    '''
    return 4*np.sin(2 * np.pi * 2e2 * t)
    # return 0

if __name__ == '__main__':
    env = simpy.Environment(0)

    runtime = 1.
    dt = 1e-6

    system = module.System(env, extern_f=external_force)

    # initial_state = np.array([0., 0.], dtype=np.float64)
    # spring_system = SpringDampingSystem(
    #     env=env,
    #     mass=7.45e-7,
    #     spring_coef=5.623,
    #     damping_coef=4.95e-6,
    #     initial_state=initial_state,
    #     runtime=runtime,
    #     dt=dt,
    #     input_force=external_force,
    # )

    # env.process(spring_system.run(runtime, dt))
    with tqdm(total=int(runtime/dt), desc='Running simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)

    plt.figure()
    plt.plot(system.spring_system.simulation_data['time'], system.spring_system.simulation_data['position'])
    plt.show()
