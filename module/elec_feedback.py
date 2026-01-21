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

class ElecFeedback(ModuleBase):
    '''
    Electrical force feedback module.
    With a working frequancy the same of the sampling rate.
    '''
    DELAY = 2e-9
    def __init__(
        self,
        env: simpy.Environment,
        system_state: SystemState,
        C0: float,
        gap: float = 0,
        v_ref: float = 0,
        duty_cycle: float = 1/2,
        runtime: float = 1,
        fs: float = 128 * 1e3,
    ) -> None:
        super().__init__(env=env, runtime=runtime+ElecFeedback.DELAY, dt=1/fs)
        self.C0 = C0
        self.gap = gap
        self.vref = v_ref
        self.vp =  self.vref
        self.vn = -self.vref
        self.fs = fs
        self.system_state = system_state
        self.duty_cycle = duty_cycle

        self._enabled = False

        # self.simulation_data = {'time': [], 'output': []}

        e0 = 8.854187817e-12
        
        self.area = self.C0 * self.gap / e0

        # F = 0.5 * C0 / d * (U_1 - U_2)**2 / (1 - x/d)**2
        # coef = 0.5 * C0 / d
        self.elec_force_coef = 0.5 * self.C0 / self.gap


    def force(self, x: float) -> float:
        '''
        Calculate the electric force.  
        With pid command controling its direction.  
        Only deliver force in duty cycle.

        Args:
            x (float): Displacement

        Returns:
            float: Electric force
        '''
        if not self._enabled:
            u_c = 0
        else:
            u_c = self.system_state.pid_cmd

        u = x / self.gap

        # Top capacitor force
        F_t = self.elec_force_coef * (self.vp - u_c)**2 / (1 - u)**2
        # Down capacitor force
        F_b = self.elec_force_coef * (u_c - self.vn)**2 / (1 + u)**2

        return F_t - F_b

    def force_const(self, x: float)-> float:
        """Electric force. 
        
        Averaged by Impulse Theorem.

        Args:
            x (float): Displacement

        Returns:
            float: Electric force
        """        
        u = x / self.gap

        # Detect stage
        # Top capacitor force
        F_t = self.elec_force_coef * (self.vp - 0)**2 / (1 - u)**2
        # Down capacitor force
        F_b = self.elec_force_coef * (0 - self.vn)**2 / (1 + u)**2

        # Feedback stage
        # Top capacitor force
        F_t_fb = self.elec_force_coef * (self.vp - self.system_state.pid_cmd)**2 / (1 - u)**2
        # Down capacitor force
        F_b_fb = self.elec_force_coef * (self.system_state.pid_cmd - self.vn)**2 / (1 + u)**2
        return self.duty_cycle * ( F_t_fb - F_b_fb ) + (1 - self.duty_cycle) * (F_t - F_b)
        

    def run(self):
        '''
        Function to be processed when simulation started.
        Update enable state every half period.
        '''
        yield self.env.timeout(ElecFeedback.DELAY)
        while self.env.now < self.runtime:
            self._enabled = True
            yield self.env.timeout(self.dt * self.duty_cycle)
            self._enabled = False
            yield self.env.timeout(self.dt * (1-self.duty_cycle))
