"""TuRBO trust-region state (Eriksson et al. 2019), used by brain_bo when BRAIN_TR=1."""
from dataclasses import dataclass

import numpy as np


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -1e30
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = int(np.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size)))


def update_state(state, Y_next):
    best_new = float(Y_next.max())
    if best_new > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1; state.failure_counter = 0
    else:
        state.success_counter = 0; state.failure_counter += 1   # +1 per ROUND (TuRBO standard), not +batch
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max); state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:      # safeguard 6: failed batch -> shrink the TR
        state.length /= 2.0; state.failure_counter = 0
    state.best_value = max(state.best_value, best_new)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state
