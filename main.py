'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys
import simpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import module
import util

matplotlib.use('TkAgg')


def external_accel(t: float):
    '''
    External Force, nothing unusual.
    '''
    return 0.4 * np.sin(2 * np.pi * 2e2 * t)
    # return 0


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a mems die.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument('--out',
                        type=str,
                        help='Data output file.',
                        default=f'data/{util.formatted_date_time}-data.json')

    return parser.parse_args()


if __name__ == '__main__':
    args = argue_parser()

    print(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        param = json.load(f)
        f.close()
        json.dump(param, sys.stdout, indent=2)

    runtime = param['runtime']
    dt = param['mechanic_dt']

    env = simpy.Environment(0)
    system = module.System(env, config=param, extern_accel=external_accel)

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

    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now < runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)

    disp = util.Signal(np.array(system.spring_system.simulation_data['position']),
                       t=np.array(system.spring_system.simulation_data['time']),
                       label='Displacement')
#   velo = util.Signal(np.array(system.spring_system.simulation_data['velocity']),
#                      t=np.array(system.spring_system.simulation_data['time']),
#                      label='Velocity')


    util.Signal.plot_all(block=False)

    out = util.Signal(np.array(system.pid.simulation_data['output']),
                      t=np.array(system.pid.simulation_data['time']),
                      label='PID out')
    out.plot_time_domain()
    out.plot_freq_domain()

    plt.show()
