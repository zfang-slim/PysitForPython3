import numpy as np

from pysit.solvers.solver_base import *
from pysit.solvers.model_parameter import *

from pysit.util.solvers import inherit_dict

__all__ = ['ConstantDensityAcousticBase']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticBase(SolverBase):
    """ Base class for solvers that use the Constant Density Acoustic Model
    (e.g., in the wave, helmholtz, and laplace domains).

    """

    _local_support_spec = {'equation_physics': 'constant-density-acoustic'}

    ModelParameters = ConstantDensityAcousticParameters

    def _compute_dWaveOp_time(self, solver_data):
        ukm1 = solver_data.km1.primary_wavefield
        uk   = solver_data.k.primary_wavefield
        ukp1 = solver_data.kp1.primary_wavefield
        return (ukp1-2*uk+ukm1)/(self.dt**2)

    def _compute_dWaveOp_frequency(self, uk_hat, nu):
        omega = (2*np.pi*nu)
        Bmat = -(omega)**2 * self.operator_components.I + \
             omega * 1j * self.operator_components.sigma_xPz + self.operator_components.sigma_xz
        return Bmat * uk_hat
        # Comment out by Zhilong
        # omega2 = (2*np.pi*nu)**2.0
        # return -1*omega2*uk_hat

    def _compute_dWaveOp_laplace(self, *args):
        raise NotImplementedError('Derivative Laplace domain operator not yet implemented.')
