'''
The PID module.
'''
import simpy
import numpy as np
from .system import SystemState


class PID:
    '''
    PID module.
    '''

    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        kp: float = -5,
        ki1: float = -0.516,
        ki2: float = -0.5,
        kd: float = 0,
        fs: float = 128 * 1e3,
        runtime: float = 1,
        target: float = 0.,
    ):
        self.env = env
        self.kp = kp
        self.ki1 = ki1
        self.ki2 = ki2
        self.kd = kd
        self.system_state = system_state
        self.fs = fs
        self.runtime = runtime
        self.target = target

        self.integral1 = 0.
        self.integral2 = 0.

        self.previous_error = 0.

        self.simulation_data = {'time': [], 'output': []}

        self.out = self.quantizer(0)

        self.env.process(self.run())

    def quantizer(self, val):
        '''
        The quantizer after PID module.
        '''
        return int(np.sign(val))

    def update(self, current_value):
        '''
        Update pid status.
        '''
        error = self.target - current_value
        self.integral2 += 0.038 * self.integral1
        feed_back = -0.01139 * self.integral2
        self.integral1 += 0.06 * feed_back + 1.55 * error
        derivative = error - self.previous_error
        self.previous_error = error

        self.out = self.quantizer(self.kp * error + self.ki1 * self.integral1 +
                                  self.ki2 * self.integral2 +
                                  self.kd * derivative)

    def run(self):
        '''
        Main loop of simulation.
        '''
        while self.env.now < self.runtime:
            self.simulation_data['time'].append(self.env.now)
            self.simulation_data['output'].append(self.out)
            self.system_state.pid_cmd = self.out
            self.update(self.system_state.mass_block_state[0])
            yield self.env.timeout(1 / self.fs)
