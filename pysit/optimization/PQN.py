

import time
import copy
from collections import deque

import numpy as np

from pysit.optimization.optimization import OptimizationBase

__all__ = ['PQN', 'LBFGS_Hessian']

__docformat__ = "restructuredtext en"

class PQN(OptimizationBase):

    def __init__(self, objective, memory_length=None, reset_on_new_inner_loop_call=True, proj_op=None, *args, **kwargs):
        OptimizationBase.__init__(self, objective, *args, **kwargs)
        self.prev_alpha = None
        self.prev_model = None
        # collections.deque uses None to indicate no length
        self.memory_length=memory_length
        self.reset_on_new_inner_loop_call = reset_on_new_inner_loop_call
        self.proj_op= proj_op

        self._reset_memory()

    def _reset_memory(self):
        self.memory = deque([], maxlen=self.memory_length)
        self._reset_line_search = True
        self.prev_model = None

    def inner_loop(self, *args, **kwargs):

        if self.reset_on_new_inner_loop_call:
            self._reset_memory()

        OptimizationBase.inner_loop(self, *args, **kwargs)

    def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
        """Compute the LBFGS update for a set of shots.

        Gives the step s as a function of the gradient vector.  Implemented as in p178 of Nocedal and Wright.

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute.
        grad : ndarray
            Gradient vector.
        i : int
            Current time index.

        """

        mem = self.memory

        q = copy.deepcopy(gradient)

        x_k = copy.deepcopy(self.base_model) #Not sure if the copy is required

        # fix the 'y' variable, since we saved the previous gradient and just
        # computed the next gradient.  That is, assume we are at k+1.  the right
        # side of memory is k, but we didn't compute y_k yet.  in the y_k slot we
        # stored a copy of the (negative) gradient.  thus y_k is that negative
        # gradient plus the current gradient.
        
        #Use the same idea for the 's' variable which is defined as the model difference. 
        #We cannot use the variable 'step' at the end of the iteration, because in inner_loop()
        #the model is updated as self.base_model += step, and the modelparameter class can
        #enforce bounds and do other types of postprocessing.
        if len(mem) > 0:
            mem[-1][2] += gradient # y
            mem[-1][1] = x_k - self.prev_model #Subtraction will result a model perturbation, which is linear.
            mem[-1][0]  = 1./mem[-1][2].inner_product(mem[-1][1]) # rho
            gamma = mem[-1][1].inner_product(mem[-1][2]) / mem[-1][2].inner_product(mem[-1][2])
        else:
            gamma = 1.0

        alphas = []

        for rho, s, y in reversed(mem):
            alpha = rho * s.inner_product(q)
            t = alpha * y
            q -= t
            alphas.append(alpha)

        alphas.reverse()

        r = gamma * q

        for alpha, m in zip(alphas, mem):
            rho, s, y = m
            beta = rho*y.inner_product(r)
            r += (alpha-beta)*s

        # Search the opposite direction
        direction = -1.0*r

        alpha0_kwargs = {'reset' : False}
        if self._reset_line_search:
            alpha0_kwargs = {'reset' : True}
            self._reset_line_search = False

        alpha = self.select_alpha(shots, gradient, direction, objective_arguments,
                                  current_objective_value=current_objective_value,
                                  alpha0_kwargs=alpha0_kwargs, **kwargs)

        self._print('  alpha {0}'.format(alpha))
        self.store_history('alpha', iteration, alpha)

        step = alpha * direction

        # these copy calls might be removable
        self.prev_model = x_k
        self.memory.append([None,None,copy.deepcopy(-1*gradient)])

        return step


    def _compute_alpha0(self, phi0, grad0, reset=False, *args, **kwargs):
        if reset:
            return phi0 / (grad0.norm()*np.prod(self.solver.mesh.deltas))**2
        else:
            return 1.0

    def _select_alpha_constraint(shots, gradient, direction, objective_arguments, current_objective_value, memory, alph0_kwargs):
        a = 1

        ## get the f(x0) and g(x0)



        ## solve the constrained quadratical problems


        ## compare the objective



class LBFGS_Hessian(object):
    def __init__(self, mem=None, gamma=1.0):
        self.mem = mem
        # shape_mem = np.shape(mem)
        # self.shape_mem = shape_mem
        self.n_mem = len(mem)

        if n_mem > 0:
            self.gamma = mem[-1][1].inner_product(mem[-1][2]) / mem[-1][2].inner_product(mem[-1][2])
        else:
            self.gamma = gamma

        if n_mem > 0:
            M1 = np.zeros((n_mem, n_mem))
            M2 = np.zeros((n_mem, n_mem))
            M4 = np.zeros((n_mem, n_mem))
            for i in range(n_mem):
                M4[i,i] = mem[i][1].inner_product(mem[i][2])
                for j in range(n_mem):
                    M1[i,j] = mem[i][1].inner_product(mem[j][1])
                    if i > j:
                        M2[i,j] = mem[i][1].inner_product(mem[j][2])
                
            
            M1 = 1.0 / gamma * M1 
            M  = np.bmat([[M1, M2],[M2.transpose(), M4]])
            self.BFGS_IM = np.linalg.inv(M)

        else:
            self.BFGS_IM = None


        

    def __mul__(self, x):
        ## Define the matrix-vector product of the l-BFGS Hessian
        sigma = 1.0 / self.gamma
        n_mem = len(mem)

        if n_mem is 0:
            output = sigma*x
        else:
            a_tmp = np.zeros((n_mem,1))
            b_tmp = np.zeros((n_mem,1))

            for i in range(n_mem):
                rho = mem[i][0]
                s = mem[i][1]
                y = mem[i][2]
                a_tmp[i] = s.inner_product(x) * sigma
                b_tmp[i] = y.inner_product(x) 
        
            c_tmp = np.concatenate((a_tmp, b_tmp))
            c_tmp = self.BFGS_IM * c_tmp

            for i in range(n_mem):
                s = mem[i][1]
                y = mem[i][2]
                if i is 0:
                    z = sigma*s*c_tmp[0] + y*c_tmp[n_mem]
                else:
                    z += sigma*s*c_tmp[i] + y*c_tmp[i+n_mem]
            
            output = sigma*x - z

        return output

    def inv(self, x):
        ## Define the inverse matrix-vector product of the l-BFGS Hessian
        
        gamma = self.gamma
        mem = self.mem

        alphas = []

        for rho, s, y in reversed(mem):
            alpha = rho * s.inner_product(x)
            t = alpha * y
            q -= t
            alphas.append(alpha)

        alphas.reverse()

        r = gamma * q

        for alpha, m in zip(alphas, mem):
            rho, s, y = m
            beta = rho*y.inner_product(r)
            r += (alpha-beta)*s
        
        return r



if __name__ == '__main__':

    import time

    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import os
    from shutil import copy2

    import sys

    from pysit import *
    from pysit.gallery import horizontal_reflector
    from pysit.gallery.layered_medium import three_layered_medium
    from pysit.util.io import *

    

    C, C0, m, d=three_layered_medium(TrueModelFileName='testtrue.mat', InitialModelFileName='testInitial.mat',
                                     initial_model_style='gradient',
                                     initial_config={'sigma': 4.0, 'filtersize': 4},)

    Nshots = 1
    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    # zpos = zmin + (1./9.)*zmax
    zpos = 0.01 * 2.0

    shots=equispaced_acquisition(m,
                                 RickerWavelet(1.0),
                                 sources=Nshots,
                                 source_depth=zpos,
                                 source_kwargs={},
                                 receivers='max',
                                 receiver_depth=zpos,
                                 receiver_kwargs={},
                                 )

    # Define and configure the wave solver
    trange=(0.0, 3.0)

    solver=ConstantDensityAcousticWave(m,
                                       spatial_accuracy_order=4,
                                       trange=trange,
                                       kernel_implementation='cpp',
                                       max_C=4.0)  # The dt is automatically fixed for given max_C (velocity)

    print(solver.max_C)

    # Generate synthetic Seismic data
    sys.stdout.write('Generating data...')
    base_model=solver.ModelParameters(m, {'C': C})
    g = base_model.Perturbation(m)

    a = 1

    # test_H = LBFGS_Hessian()
    # b = np.mat([0,1])
    # b = b.transpose()
    # c = test_H.inv(b)
    # d = test_H * b
    # e = b*test_H
    # print('c=', c)
    # print('d=', d)




