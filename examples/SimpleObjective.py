import numpy as np
import copy

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.frequency_modeling import FrequencyModeling

__all__ = ['SimpleExponetialObjective']

__docformat__ = "restructuredtext en"


class SimpleExponetialObjective(ObjectiveFunctionBase):

    def __init__(self, solver=None, parallel_wrap_shot=ParallelWrapShotNull()):
        self.parallel_wrap_shot = parallel_wrap_shot
        self.solver = solver

    def evaluate(self, shots, m0):
        norm_m = np.linalg.norm(m0.data)
        norm_m2 = norm_m**2.0
        obj_value = np.exp(-0.5 * norm_m2)

        return obj_value

    def compute_gradient(self, shots, m0, aux_info):
        norm_m = np.linalg.norm(m0.data)
        norm_m2 = norm_m**2.0
        grad = -np.exp(-0.5 * norm_m2) * m0

        return grad
