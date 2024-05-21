import simpy
import numpy as np

class PID:
    def __init__(
        self,
        env: simpy.Environment,
        kp: float,
        ki1: float,
        ki2: float,
        kd: float,
        target: float = 0.,
    ):
        self.env = env
        self.kp = kp
        self.ki1 = ki1
        self.ki2 = ki2
        self.kd = kd
        self.target = target

        self.integral1 = 0.
        self.integral2 = 0.

        self.previous_error = 0.

        self.simulation_data = {'time': [], 'output': []}

        self.out = self.quantizer(0)

    def quantizer(self, val):
        return int(np.sign(val))

    def update(self, current_value):
        error = self.target - current_value
        self.integral2 += 0.038 * self.integral1
        feed_back = -0.01139 * self.integral2
        self.integral1 += 0.06 * feed_back + 1.55 * error
        derivative = error - self.previous_error
        self.previous_error = error

        self.out = self.quantizer(self.kp * error + self.ki1 * self.integral1 + self.ki2 * self.integral2 + self.kd * derivative)
        return self.out

    def run(self, runtime, dt):
        pass

