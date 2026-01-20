'''
Module for electrical force feedback.
'''
# import numpy as np
import simpy
from .system import SystemState
from .base import ModuleBase

class C2V:
    """Displacement to capacitance changement to voltage changement.
    """    
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        param: dict[str, float],
        ) -> None:
        self.env = env
        self.system_state = system_state

        self.C0 = param['C0']
        self.gap = param['gap']
        self.Cf = param['Cf']
        self.vref = param['v_ref']
        self.vp =  self.vref
        self.vn = -self.vref

        # C to V sensitivity
        self.k_cv = (self.vp - self.vn) / self.Cf

    def x2c2v(self) -> float:
        x = self.system_state.get_displacement()
        u = x / self.gap

        # Top capacitor and down capacitor
        C_t = self.C0 / (1 - u)
        C_b = self.C0 / (1 + u)
        # Capacitance change
        delta_C = C_t - C_b
        # voltage change
        delta_V = self.k_cv * delta_C
        
        return delta_V

class ElecFeedback:
    '''
    Electrical force feedback module.
    With a working frequancy the same of the sampling rate.
    '''

    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        area: float = 0,
        gap: float = 0,
        v_ref: float = 0,
        runtime: float = 1,
        fs: float = 128 * 1e3,
    ) -> None:
        self.env = env
        self.area = area
        self.gap = gap
        self.v_ref = v_ref
        self.runtime = runtime
        self.fs = fs
        self.system_state = system_state

        self._enabled = False

        # self.simulation_data = {'time': [], 'output': []}

        e0 = 8.854187817e-12

        self.elec_force_coef = 0.5 * self.area * e0 * (2 * self.v_ref)**2

        self.env.process(self.run(self.fs))

    def force(self) -> float:
        '''
        Calculate the electric force.
        With pid command controling its direction.
        Only deliver force in half of the time.
        '''
        if not self._enabled:
            return 0

        disp = self.system_state.mass_block_state[0]

        if self.system_state.pid_cmd == 1:
            # pull up
            distance = self.gap - disp
            coef = self.elec_force_coef
        else:  # self.system_state.pid_cmd == -1
            # pull down
            distance = self.gap + disp
            coef = -self.elec_force_coef

        return coef / (distance**2)

    def run(self, fs):
        '''
        Function to be processed when simulation started.
        Update enable state every half period.
        '''
        periode = 1 / fs
        while self.env.now < self.runtime:
            self._enabled = False
            yield self.env.timeout(periode / 2)
            self._enabled = True
            yield self.env.timeout(periode / 2)
