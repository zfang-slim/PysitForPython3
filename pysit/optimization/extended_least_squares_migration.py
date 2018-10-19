

import time
import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import cg, gmres

from pysit.optimization.optimization import OptimizationBase
from pysit.solvers.model_parameter import *

__all__ = ['ExtendLSM']

__docformat__ = "restructuredtext en"


class ExtendLSM(object):

    def __init__(self, objective, shots, m0, simdata, max_sub_offset, h, imaging_period=None, frequencies=None, krylov_maxiter=20, *args, **kwargs):
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

    def run_lsm(self, input=None):
        """run extended least squares reverse time migration

        """

        krylov_maxiter = self.krylov_maxiter

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

        rhs.data = rhs.data * np.prod(m0.mesh.deltas)

        rhs = rhs.data.reshape(-1)

        m1_extend = ExtendedModelingParameter2D(m0.mesh, self.max_sub_offset, self.h)
        if input is not None:
            x0 = input
        else:
            x0 = m1_extend.data.reshape(-1)

        def matvec(x):
            m1_extend.setter(x0)
            if self.solver.supports['equation_dynamics'] == "time":
                linfwdret = self.tools.linear_forward_model_extend(self.shots,
                                                                   m0,
                                                                   m1_extend,
                                                                   self.max_sub_offset,
                                                                   self.h,
                                                                   ['simdata']
                                                                   )
                lindatas = linfwdret['simdata']

                m1_out = self.tools.migrate_shots_extend(self.shots, m0, lindatas,
                                                         self.max_sub_offset, self.h,
                                                         self.imaging_period
                                                         )
            else:
                linfwdret = tools.linear_forward_model_extend(self.shots,
                                                              m0,
                                                              m1_extend,
                                                              self.frequencies,
                                                              self.max_sub_offset,
                                                              self.h,
                                                              ['simdata']
                                                              )
                lindatas = linfwdret['simdata']
                for i in range(len(shots)):
                    lindatas[i] = lindatas[i] * self.tool.solver.dt

                m1_out = self.tools.migrate_shots_extend(self.shots, m0, lindatas,
                                                         self.frequencies,
                                                         self.max_sub_offset, self.h,
                                                         return_parameters=['imaging_condition']
                                                         )

            m1_out.data = m1_out.data * np.prod(m0.deltas)
            return m1_out

        A_shape = (len(rhs), len(rhs))

        A = LinearOperator(shape=A_shape, matvec=matvec, dtype=rhs.dtype)

        resid = []

#       d, info = cg(A, rhs, maxiter=self.krylov_maxiter, residuals=resid)
        x_out, info = gmres(A, rhs, x0=x0, maxiter=self.krylov_maxiter, residuals=resid)
        m1_extend.setter(x_out)
        self.m_out = m1_extend

        return m1_extend
