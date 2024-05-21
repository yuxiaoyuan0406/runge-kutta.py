'''
Author: Yu Xiaoyuan
'''
import argparse
import simpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import module
import yaml

matplotlib.use('TkAgg')

def external_force(t: float):
    '''
    External Force, nothing unusual.
    '''
    return 4*np.sin(2 * np.pi * 2e2 * t)
    # return 0

def argue_parser():
    parser = argparse.ArgumentParser(description='Run simulation of a mems die.')

    parser.add_argument('--param', type=str,
        help='An optional simulation parameters file in yaml format',
        default='parameters/default.yaml')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argue_parser()

    print(f'Using parameters from file `{args.param}`.')
    with open(args.param) as f:
        param = yaml.safe_load(f)
        f.close()
        print(yaml.dump(param))

    runtime = param['runtime']
    dt = param['mechanic_dt']

    env = simpy.Environment(0)
    system = module.System(env, config=param, extern_f=external_force)

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
