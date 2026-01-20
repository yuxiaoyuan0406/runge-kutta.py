'''
Author: Yu Xiaoyuan
'''
import argparse
import json
import sys

import simpy
import numpy as np
import matplotlib
import os
from typing import Callable
from tqdm import tqdm

sys.path.append('.')
import module
from module import SpringDampingSystem
from module import SystemState
from module import C2V
from module import ElecFeedback
import util

# matplotlib.use('TkAgg')
module.spring_damping.USING_CPP_BACKEND = True


def argue_parser():
    '''
    Arguements.
    '''
    parser = argparse.ArgumentParser(
        description='Run simulation of a spring-damping system with pid feedback.')

    parser.add_argument(
        '--param',
        type=str,
        help='An optional simulation parameters file in json format',
        default='parameters/default.json')
    parser.add_argument('--out',
                        type=str,
                        help='Data output directory.',
                        default=f'data/{util.formatted_date_time}-spr-dmp-pid-test')
    parser.add_argument('--plot',
                        type=str,
                        help='Plot data from directory. If not None, simulation will not be executed.',
                        default=None)
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='whether to save the simulation result')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Print extra info')
    parser.add_argument('--show',
                        action='store_true',
                        default=False,
                        help='Plot result')

    return parser.parse_args()


class PID(module.ModuleBase):
    PID_DELAY = 1e-9
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        fs: float = 128 * 1e3,
        runtime: float = 1.,
        error_getter: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(env=env, runtime=runtime+PID.PID_DELAY, dt=1/fs)
        self.system_state = system_state
        self.fs = fs

        if error_getter is None:
            error_getter = system_state.get_displacement
        self.error_getter = error_getter

        self.a = [9., 28., 1.667, 0.1667, 0.0093]
        # self.a = [9., 28., 1.667, 0., 0.]
        self.b = [5.051e-5, 0.005]

        self.inter0 = 0.
        self.inter1 = 0.
        self.inter2 = 0.

        self.error = 0.
        self.out = 0.

        self.simulation_data = {'time': [], 'output': []}

    def update(self, current_value):
        error_pre = self.error
        self.error = current_value
        diff = self.error - error_pre
        inter2_pre = self.inter2
        inter1_pre = self.inter1
        inter0_pre = self.inter0
        self.inter2 += inter1_pre - self.b[0] * inter2_pre
        self.inter1 += inter0_pre - self.b[1] * inter2_pre
        self.inter0 += error_pre

        self.out = \
            self.a[0] * self.error + self.a[1] * diff    \
            + self.a[2] * self.inter0 + self.a[3] * self.inter1 \
            + self.a[4] * self.inter2

    def save_state(self):
        self.simulation_data['time'].append(self.env.now - PID.PID_DELAY)
        self.simulation_data['output'].append(self.out)

    def run(self):
        yield self.env.timeout(PID.PID_DELAY)
        while self.env.now < self.runtime:
            # self.system_state.pid_cmd = self.out
            self.update(self.error_getter())
            self.save_state()
            yield self.env.timeout(self.dt)

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'time'),
                np.array(self.simulation_data['time']))
        np.save(os.path.join(directory, 'pid'),
                np.array(self.simulation_data['output']))

    @classmethod
    def load_from_file(cls, directory):
        '''
        Load the simulation data from a file.
        '''
        t = np.load(os.path.join(directory, 'time.npy'))
        pid = np.load(os.path.join(directory, 'pid.npy'))

        pid = util.Signal(pid, t=t, label='PID out')
        return pid

class TopSystem(module.ModuleBase):
    def __init__(self, env: simpy.Environment,
                 param: dict,
                 runtime: float, fs: float) -> None:
        super().__init__(env=env, runtime=runtime, dt=1/fs)
        self.fs = fs
        self.system_state = SystemState()

        self.mechanic_dt = param['mechanic_dt']
        self.initial_state = np.array(
            param.get('initial_state', [0,0]),
            dtype=np.float64
        )
        self.spring_system = SpringDampingSystem(
            env=self.env,
            mass=param['mass'],
            spring_coef=param['spring_coef'],
            damping_coef=param['damping_coef'],
            initial_state=self.initial_state,
            system_state=self.system_state,
            runtime=self.runtime,
            dt=self.mechanic_dt,
            input_accel=self.__unit_step
        )

        self.cv = C2V(
            env=env,
            system_state=self.system_state,
            param=param
        )

        self.pid = PID(
            env=env, 
            system_state=self.system_state, 
            fs=self.fs, 
            runtime=self.runtime,
            error_getter=self.cv.x2c2v
        )
    
    def __unit_pulse(self, t):
        if 0. <= t and t < self.dt:
            return 1 / self.dt
        else:
            return 0.
    
    def __unit_step(self, t):
        if t < 0.001:
            return 0.
        else:
            return 0.1*9.81
    
    def __default_sin(self, t):
        return np.sin(2*np.pi*1e3*t)
    
    def run(self):
        '''
        '''
        yield self.env.timeout(self.pid.PID_DELAY)
        while self.env.now < self.runtime:
            yield self.env.timeout(self.dt)

    def save(self, directory):
        '''
        Save the simulation data to a file.
        '''
        os.makedirs(directory, exist_ok=True)
        self.pid.save(directory=os.path.join(directory, 'pid'))
        self.spring_system.save(directory=os.path.join(directory, 'mass_block'))
        os.makedirs(os.path.join(directory, 'matlab'), exist_ok=True)


    @classmethod
    def load_from_file(cls, directory):
        '''
        Load the simulation data from a file.
        '''
        class RetClass:
            def __init__(self) -> None:
                self.displacement: util.Signal = None # type: ignore
                self.pid: util.Signal = None # type: ignore
        
        posi, velo = SpringDampingSystem.load_from_file(os.path.join(directory, 'mass_block'))
        pid = PID.load_from_file(os.path.join(directory, 'pid'))
        ret = RetClass()
        ret.displacement = posi
        ret.pid = pid
        return ret
        



def main(args):
    verbose = args.verbose

    if verbose:
        print(f'Using parameters from file `{args.param}`.')
    with open(args.param, 'r', encoding='utf-8', errors='replace') as f:
        # param = json.load(f)
        param = util.SpringDampingParameters(f)
        f.close()

    param['runtime'] = 0.01
    runtime = param['runtime']
    fs = param['samp_rate']

    if verbose:
        print(json.dumps(param, indent=2))

    env = simpy.Environment(0)
    top = TopSystem(
        env=env,
        param=param,
        runtime=runtime,
        fs=fs)
    
    time_slice = 1000
    with tqdm(total=time_slice, desc='Running') as pbar:
        while env.now <= runtime:
            env.run(until=env.now + runtime / time_slice)
            pbar.update(1)

    # Create a data object for analysis
    pid_out = util.Signal(np.array(top.pid.simulation_data['output']),
                          t=np.array(top.pid.simulation_data['time']),
                          label='PID out')
    disp = util.Signal(np.array(top.spring_system.simulation_data['position']),
                       t=np.array(top.spring_system.simulation_data['time']),
                       label='Displacement')

    # Plot result
    if args.show:
        fig_time, ax_time = util.default_time_plot_fig()
        pid_out.plot_time_domain(ax=ax_time, show=True, block=False)
        fig_time, ax_time = util.default_time_plot_fig()
        disp.plot_time_domain(ax=ax_time, show=True, block=True)


    if args.save:
        top.save(args.out)

def plot(args):
    data_path = args.plot
    data = TopSystem.load_from_file(data_path)
    
    matlab_data_path = os.path.join(data_path, 'matlab', 'pid.h5')
    import h5py
    with h5py.File(matlab_data_path, "r") as f:
        matlab_time = np.ravel(f["/t"][...])
        matlab_out = np.array(f["/y"][...])
        matlab_time = np.asarray(matlab_time).reshape(-1)
        matlab_out = np.asarray(matlab_out).reshape(-1)

    matlab_pid = util.Signal(matlab_out,
                       t=matlab_time,
                       color='red',
                       linestyle=':',
                       label='MATLAB')

    fig_disp, ax_time = util.default_time_plot_fig()
    data.displacement.plot_time_domain(ax=ax_time, show=True)
    
    fig_pid, ax_time = util.default_time_plot_fig()
    data.pid.plot_time_domain(ax=ax_time, show=False)
    matlab_pid.plot_time_domain(ax=ax_time, show=True, block=True)

    fig_pid.savefig(os.path.join(data_path, 'pid-time.png'),
                    bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    args = argue_parser()
    if args.plot == None:
        main(args)
    else:
        plot(args)
