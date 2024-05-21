import numpy as np
import simpy

class ElecFeedback:
    def __init__(
        self,
        env: simpy.Environment,
        area: float = 0,
        gap: float = 0,
        v_ref: float = 0,
    ) -> None:
        self.env = env
        self.area = area
        self.gap = gap
        self.v_ref = v_ref

        self._enabled = False

        e0 = 8.854187817e-12

        self.elec_force_coef = 0.5 * e0 * (2 * self.v_ref) ** 2 / 10

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def force(self, disp: float, cmd: int) -> float:
        return 0
