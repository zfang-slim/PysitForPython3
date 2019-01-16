

import time
import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import cg, gmres

from pysit.optimization.optimization import OptimizationBase
from pysit.solvers.model_parameter import *
from pysit.util.parallel import *
from mpi4py import MPI

__all__ = ['ExtendLSM']

__docformat__ = "restructuredtext en"


class ExtendLSM(object):

    def __init__(self, objective, shots, m0, simdata, max_sub_offset, h, 
                 imaging_period=None, frequencies=None, krylov_maxiter=20, 
                 weight_matrix=None, regularization_value=None, parallel_wrap_shot=ParallelWrapShot(),
                 *args, **kwargs):
        self.tools = objective
        self.m0 = m0
        self.shots = shots
        self.simdata = simdata
        self.max_sub_offset = max_sub_offset
        self.h = h
        self.imaging_period = imaging_period
        self.frequencies = frequencies
        self.krylov_maxiter = krylov_maxiter
        self.m_out = []
        self.weight_matrix = weight_matrix
        self.regularization_value = regularization_value
        self.parallel_wrap_shot = parallel_wrap_shot
        if parallel_wrap_shot.size == 1: 
            self.parallel_wrap_shot.use_parallel = False

    def set_m0(self, m0):
        self.m0 = m0

    def run_lsm(self, input=None):
        """run extended least squares reverse time migration

        """

        krylov_maxiter = self.krylov_maxiter
        weight_matrix = self.weight_matrix
        regularization_value = self.regularization_value

        m0 = self.m0

        if self.tools.solver.supports['equation_dynamics'] == "time":
            rhs = self.tools.migrate_shots_extend(self.shots, m0, self.simdata,
                                                  self.max_sub_offset, self.h,
                                                  self.imaging_period
                                                  )
        else:
            rhs = self.tools.migrate_shots_extend(self.shots, m0, self.simdata,
                                                  self.frequencies,
                                                  self.max_sub_offset, self.h,
                                                  return_parameters=['imaging_condition'])

        rhs.data = rhs.data * np.prod(m0.mesh.deltas) / self.tools.solver.dt

        rhs = rhs.data.reshape(-1)

        if self.parallel_wrap_shot.use_parallel:
                rhs_global = self.parallel_wrap_shot.comm.allreduce(rhs, op=MPI.SUM)
                rhs = rhs_global

        m1_extend = ExtendedModelingParameter2D(m0.mesh, self.max_sub_offset, self.h)
        if input is not None:
            x0 = input.data.reshape(-1)
        else:
            x0 = m1_extend.data.reshape(-1)

        def matvec(x):
            m1_extend.setter(x)
            if self.tools.solver.supports['equation_dynamics'] == "time":
                linfwdret = self.tools.linear_forward_model_extend(self.shots,
                                                                   m0,
                                                                   m1_extend,
                                                                   self.max_sub_offset,
                                                                   self.h,
                                                                   ['simdata']
                                                                   )
                lindatas = linfwdret['simdata']

                for i in range(len(self.shots)):
                    lindatas[i] = lindatas[i] #* self.tools.solver.dt

                m1_out = self.tools.migrate_shots_extend(self.shots, m0, lindatas,
                                                         self.max_sub_offset, self.h,
                                                         self.imaging_period
                                                         )
                m1_out.data = m1_out.data * np.prod(m0.mesh.deltas) / self.tools.solver.dt
                    #/ self.tools.solver.dt
                

            else:
                linfwdret = self.tools.linear_forward_model_extend(self.shots,
                                                                   m0,
                                                                   m1_extend,
                                                                   self.frequencies,
                                                                   self.max_sub_offset,
                                                                   self.h,
                                                                   ['simdata']
                                                                   )
                lindatas = linfwdret['simdata']
                

                m1_out = self.tools.migrate_shots_extend(self.shots, m0, lindatas,
                                                         self.frequencies,
                                                         self.max_sub_offset, self.h,
                                                         return_parameters=['imaging_condition']
                                                         )

                m1_out.data = m1_out.data * np.prod(m0.mesh.deltas) 

            if weight_matrix is not None:
                if regularization_value is None:
                    raise TabError('A weight_matrix is passed, but not regularization_value is given. Please give a value to regularization_value')
                else:
                    if weight_matrix=='linear_h':
                        sh_data = m1_out.sh_data
                        max_sub_offset = m1_out.max_sub_offset
                        weight_h = np.linspace(-max_sub_offset, max_sub_offset, sh_data[1])
                        for i in range(sh_data[1]):
                            m1_out.data[:, i] += regularization_value * weight_h[i]**2.0 * \
                                                 m1_extend.data[:, i] / self.parallel_wrap_shot.size * \
                                                 np.prod(m0.mesh.deltas)
    
            xout_local = np.reshape(m1_out.data, (np.prod(m1_out.sh_data), 1))

            if self.parallel_wrap_shot.use_parallel:
                xout_global=self.parallel_wrap_shot.comm.allreduce(xout_local, op=MPI.SUM)
            else:
                xout_global = xout_local

            return xout_global

        A_shape = (len(rhs), len(rhs))

        A = LinearOperator(shape=A_shape, matvec=matvec, dtype=rhs.dtype)

        resid = []
        rhs = np.reshape(rhs,(len(rhs),1))
        x0 = np.reshape(x0,(len(x0),1))


#       d, info = cg(A, rhs, maxiter=self.krylov_maxiter, residuals=resid)
        x_out, info = cg(A, rhs, x0=x0, maxiter=self.krylov_maxiter, residuals=resid)
        m1_extend.setter(x_out)
        self.m_out = m1_extend

        return m1_extend
