

import time
import copy
import sys

import numpy as np

from pysit.optimization.optimization import OptimizationBase

__all__=['PGD']

__docformat__ = "restructuredtext en"

class PGD(OptimizationBase):

    def __init__(self, objective, proj_op=None, *args, **kwargs):
        OptimizationBase.__init__(self, objective, *args, **kwargs)
        self.prev_alpha = None
        self.proj_op = proj_op

    def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
        """Compute the adjustment of the gradient step for a set of shots.

        Gives the step s as a scalar multiple of the gradient vector.

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute.
        grad : ndarray
            Gradient vector.
        i : int
            Current time index.

        """

        # search the negative gradient
        direction = -1*gradient

        alpha0_kwargs = {'reset' : False}
        if iteration == 0:
            alpha0_kwargs = {'reset' : True}

        alpha = self.select_alpha(shots, gradient, direction, objective_arguments,
                                  current_objective_value=current_objective_value,
                                  alpha0_kwargs=alpha0_kwargs, **kwargs)

        self._print('  alpha {0}'.format(alpha))
        self.store_history('alpha', iteration, alpha)

        if self.proj_op is None:
            step = alpha * direction
        else:
            step = self.proj_op(self.base_model + alpha * direction) - self.base_model

        return step

    def _compute_alpha0(self, phi0, grad0, reset=False, upscale_factor=None, **kwargs):
        if reset or (self.prev_alpha is None):
            print('Works')
            return 0.0001 * np.sqrt(self.base_model.inner_product(self.base_model) / grad0.inner_product(grad0))
        else:
            return self.prev_alpha / upscale_factor
