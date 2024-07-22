'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

sys.path.append('.')
import simpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from module import SpringDampingSystem
from module import SystemState
import util

matplotlib.use('TkAgg')


def external_accel(t):
    '''
    External Force, nothing unusual.
    '''
    return 4e-1 * np.sin(2 * np.pi * 5e1 * t)


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
    # runtime = .01
    dt = param['mechanic_dt']
    # dt = 1e-8

    env = simpy.Environment(0)
    # system = module.System(env, config=param, extern_accel=external_accel)

    initial_state = np.array(param['initial_state'], dtype=np.float64)
    spring_system = SpringDampingSystem(
        env=env,
        system_state=SystemState(),
        mass=param['mass'],
        spring_coef=param['spring_coef'],
        damping_coef=param['damping_coef'],
        initial_state=initial_state,
        runtime=runtime,
        dt=dt,
        input_accel=external_accel,
    )

    # env.process(spring_system.run(runtime, dt))
    with tqdm(total=int(runtime / dt), desc='Running simulation') as pbar:
        while env.now < runtime:
            env.run(until=env.now + dt)
            pbar.update(1)

    _, t_ax = util.plot(np.array(spring_system.simulation_data['time']),
                        np.array(spring_system.simulation_data['position']),
                        label='runge-kutta')

    power, phase = util.freq_and_plot(
        np.array(spring_system.simulation_data['position']),
        dt,
        label='runge-kutta',
        log=True,
    )

    def transfer_func(f):
        w = 2 * np.pi * f
        jw = (1j) * w
        k = param['spring_coef']
        b = param['damping_coef']
        m = param['mass']
        return 1 / (jw**2 + jw * b / m + k / m)

    f, df, input_freq = util.t_to_f(external_accel(
        np.array(spring_system.simulation_data['time'])),
                                    dt,
                                    retstep=True)
    h = transfer_func(f)
    output_freq = h * input_freq
    _, output = util.f_to_t(output_freq, df, retstep=False)

    power.plot(f,
               20 * np.log10(np.abs(output_freq)),
               label='transfer function')
    phase.plot(f, np.unwrap(np.angle(output_freq)), label='transfer function')
    t_ax.plot(np.array(spring_system.simulation_data['time']),
              np.real(output) + np.imag(output),
              label='transfer function')

    t_ax.legend(loc='upper right')
    power.legend(loc='upper right')
    phase.legend(loc='upper right')
    t_ax.grid()
    power.grid()
    phase.grid()

    plt.show()

    simulation_data = {
        'parameters': {},
        'mass_block_state': {},
        'quantized_output': {}
    }
    simulation_data['parameters'] = param
    simulation_data['mass_block_state'] = spring_system.simulation_data
    # simulation_data['quantized_output'] = system.pid.simulation_data

    util.save(args.out, simulation_data)
