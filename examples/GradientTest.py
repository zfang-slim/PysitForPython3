
import numpy as np

__all__ = ['TemporalLeastSquares']

__docformat__ = "restructuredtext en"


class GradientTest(object):

    """Class for Gradient test"""

    def __init__(self, objective, model_perturbation=None, length_ratio=None):
        """Construct for the GradientTest class

        Parameters
        ----------
        model_perturbation : a perturbation direction for model parameter, we
        analyze the gradient behavior on this direction
        length_ratio : a step size vector to identify different step size

        """

        self.objective_function = objective
        self.solver = objective.solver
        self.use_parallel = objective.use_parallel()
        self.model_perturbation = model_perturbation
        self.length_ratio = length_ratio
        self.objective_value = None
        self.first_order_difference = None
        self.zero_order_difference = None
        self.base_model = None

    def __call__(self, shots):
        aux_info = {'objective_value': (True, None),
                    'residual_norm': (True, None)}
        model_perturbation = self.model_perturbation
        length_ratio = self.length_ratio
        n_ratio = len(length_ratio)
        # self.base_model = self.solver.ModelParameters(self.solver.mesh)
        # Compute the gradient
        gradient = self.objective_function.compute_gradient(
            shots, self.base_model, aux_info=aux_info)
        objective_value_original = aux_info['objective_value'][1]

        for i in range(0, n_ratio):
            ratio_i = length_ratio[i]
            model = self.base_model + ratio_i * model_perturbation

            fi0 = self.objective_function.evaluate(shots,
                                                   model)
            fi1 = fi0 + np.dot(ratio_i * model_perturbation, gradient)
            diff_f0 = abs(objective_value_original - fi0)
            diff_f1 = abs(objective_value_original - fi1)
            self.objective_value.append(fi0)
            self.objective_zero_order_difference.append(diff_f0)
            self.objective_first_order_difference.append(diff_f1)
