

import time
import copy
from collections import deque

import numpy as np

from pysit.optimization.optimization import OptimizationBase

__all__ = ['PQN', 'LBFGS_Hessian']

__docformat__ = "restructuredtext en"

class PQN(OptimizationBase):

    def __init__(self, objective, memory_length=None, 
                 reset_on_new_inner_loop_call=True, 
                 proj_op=None, maxiter_PGD=1000, 
                 maxiter_linesearch_PGD=100, 
                 *args, **kwargs):
        OptimizationBase.__init__(self, objective, *args, **kwargs)
        self.prev_alpha = None
        self.prev_model = None
        # collections.deque uses None to indicate no length
        self.memory_length=memory_length
        self.reset_on_new_inner_loop_call = reset_on_new_inner_loop_call
        self.proj_op=proj_op
        self.maxiter_PGD=maxiter_PGD
        self.maxiter_linesearch_PGD=maxiter_linesearch_PGD

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

        # alphas = []

        # for rho, s, y in reversed(mem):
        #     alpha = rho * s.inner_product(q)
        #     t = alpha * y
        #     q -= t
        #     alphas.append(alpha)

        # alphas.reverse()

        # r = gamma * q

        # for alpha, m in zip(alphas, mem):
        #     rho, s, y = m
        #     beta = rho*y.inner_product(r)
        #     r += (alpha-beta)*s

        # # Search the opposite direction
        # direction = -1.0*r

        ## Use Projected Gradient descent method to solve the constrained quadratic optimization problem
        if len(mem) > 0:
            H_BFGS = LBFGS_Hessian(mem)
        else:
            gamma0 = 0.0001 * np.sqrt(x_k.inner_product(x_k) / gradient.inner_product(gradient))
            H_BFGS = LBFGS_Hessian(mem, gamma=gamma0)

        proj_op = self.proj_op
        quadratic_obj = Quadratic_obj(gradient, H_BFGS, x_k)
        PGDsolver = ProjectedGradientDescent(quadratic_obj, proj_op)
        if proj_op is not None:
            # initial_value_PGD = proj_op(x_k + H_BFGS.inv(-1.0*gradient))
#             initial_value_PGD = self._compute_initial_PGD(x_k, H_BFGS.inv(-1.0*gradient), proj_op)
            initial_value_PGD = x_k
            x_kp1, f_history, g_history, x_history = PGDsolver(
                self.maxiter_PGD, self.maxiter_linesearch_PGD, initial_value_PGD, verbose=False)
            direction = x_kp1 - x_k
        else:
            direction = H_BFGS.inv(-1.0*gradient)
            x_kp1 = x_k + direction

        

        alpha0_kwargs = {'reset' : False}
        if self._reset_line_search:
            alpha0_kwargs = {'reset' : True}
            self._reset_line_search = False

        alpha0_kwargs = {'reset': False}

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

    def _compute_initial_PGD(self, x, dx, proj_op):
        alpha = 0.5
        stop = False
        while stop is not True:
            # y = proj_op(x + dx)
            y = x + dx
            if np.isfinite(np.linalg.norm(y.data)) == 1:
                stop = True 
            else:
                dx = alpha * dx

        return y


    def _compute_alpha0(self, phi0, grad0, reset=False, *args, **kwargs):
        if reset:
            return phi0 / (grad0.norm()*np.prod(self.solver.mesh.deltas))**2
        else:
            return 1.0


class Quadratic_obj(object):
    def __init__(self, first_order_term, second_order_term, x_org):
        self.first_order_term = first_order_term
        self.second_order_term = second_order_term
        self.x_org = x_org

    def __call__(self,x):
        dx = x - self.x_org
        obj = self.first_order_term.inner_product(dx) + 0.5 * dx.inner_product(self.second_order_term * dx)
        gradient = self.first_order_term + self.second_order_term * dx
        Hessian = self.second_order_term

        return obj, gradient, Hessian



class ProjectedGradientDescent(object):
    def __init__(self, objective, proj_op):
        self.objective_function = objective
        self.proj_op = proj_op
        
    def __call__(self, maxiter, maxiter_linesearch, initial_value, verbose=False):
       
        x0 = initial_value
        objective = self.objective_function
        proj_op = self.proj_op
        xk = x0
        fk, gk, Hk = objective(xk)
        f_history = []
        g_history = []
        x_history = []
        f_history.append(fk)
        g_history.append(np.linalg.norm(gk.data))
        x_history.append(np.linalg.norm(xk.data))

        if verbose is True:
            print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', 'f', '|g|', '|x|'))
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(0, fk, g_history[0], x_history[0]))


        stop = False

        itercnt = 0

        while not stop:
            if itercnt == 0:
                alpha = None

            xkp1, fkp1, gkp1, alpha = self._backtrack_line_search_PGD(objective, gk, xk, alpha)
            df = fkp1 - fk
            if fkp1 > fk or np.abs(fkp1-fk) / np.abs(fk)<10.0**(-6):
                stop = True
            else:
                itercnt += 1
                if itercnt > maxiter:
                    stop = True
                else:
                    xk   = xkp1
                    fk   = fkp1
                    gk   = gkp1

                    f_history.append(fk)
                    g_history.append(np.linalg.norm(gk.data))
                    x_history.append(np.linalg.norm(xk.data))

                    if verbose is True:
                        print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}  {4: 3.6f}    {5: 3.6f}'.format(
                            itercnt, fk, g_history[-1], x_history[-1], alpha, df))

        
        return xk, f_history, g_history, x_history

    def _backtrack_line_search_PGD(self, objective, gradient, initial_value, alpha0=None):

        if alpha0 is None:
            if initial_value is None:
                alpha = 1.0
            else:
                alpha = 0.01 * np.sqrt(initial_value.inner_product(initial_value) / gradient.inner_product(gradient))
        else:
            alpha = alpha0


        xk = initial_value
        geom_fac = 0.5
        geom_fac_up = 0.7
        goldstein_c = 1e-6  # 1e-4
        max_linesearch_iterations_PGD = 100

        fp_comp = 1e-8

        fk, _, _, = self.objective_function(xk)

        stop = False 
        itercnt = 1
        
        while not stop:
            xkptmp = xk - alpha * gradient
            xkptmp = self.proj_op(xkptmp)
            fkp1, gkp1, _, = self.objective_function(xkptmp)

            cmpval = fk + alpha * goldstein_c * gradient.inner_product((-alpha)*gradient)
            if (fkp1 <= cmpval) or ((abs(fkp1-cmpval)/abs(fkp1)) <= fp_comp):
                stop = True
            elif itercnt > max_linesearch_iterations_PGD:
                stop = True
            else:
                itercnt += 1
                alpha = alpha * geom_fac

        return xkptmp, fkp1, gkp1, alpha




## The following two classes are for testing, we should remove them later or move them to somewhere else

class BoundProjection(object):
    def __init__(self, lowerbound, upperbound):
        self.lbound = lowerbound
        self.ubound = upperbound

    def __call__(self, x):

        y = copy.deepcopy(x)
        for i in range(len(x.data)):
            if y.data[i] < self.lbound:
                y.data[i] = self.lbound

            if y.data[i] > self.ubound:
                y.data[i]=self.ubound

        return y


class LineProjection(object):
    def __init__(self, fakeinput=None):
        fakeinput = None

    def __call__(self, x):

        y = copy.deepcopy(x)

        if y.data[0]+y.data[1]+ 1.0 < 0:
            x_tmp = (y.data[0]-1.0-y.data[1]) / 2.0
            y_tmp = -x_tmp - 1.0
            y.data[0] = x_tmp
            y.data[1] = y_tmp

        return y
            

class LBFGS_Hessian(object):
    def __init__(self, memory=None, gamma=1.0):
        self.memory = memory
        # shape_mem = np.shape(mem)
        # self.shape_mem = shape_mem
        n_mem = len(memory)
        self.n_mem = n_mem
        mem = memory

        if n_mem > 0:
            gamma = mem[-1][1].inner_product(mem[-1][2]) / mem[-1][2].inner_product(mem[-1][2])
            self.gamma = gamma
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
            M  = np.bmat([[M1, M2],[M2.transpose(), -M4]])
            self.BFGS_IM = np.linalg.inv(M)

        else:
            self.BFGS_IM = None


        

    def __mul__(self, x):
        ## Define the matrix-vector product of the l-BFGS Hessian
        mem = self.memory
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
                    z = sigma*np.asscalar(c_tmp[0]) * s \
                        + np.asscalar(c_tmp[n_mem]) * y
                else:
                    z += sigma*np.asscalar(c_tmp[i]) * s \
                        + np.asscalar(c_tmp[i+n_mem]) * y
            
            output = sigma*x - z

        return output

    def inv(self, x):
        ## Define the inverse matrix-vector product of the l-BFGS Hessian
        
        gamma = self.gamma
        mem = self.memory

        alphas = []
        q = copy.deepcopy(x)

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

    n_mem = 1

    pmlz = PML(0, 100, ftype='quadratic')

#   pmlz = Dirichlet()

    z_config = (0.1, 0.11, pmlz, Dirichlet())
    z_config = (0.1, 0.11, pmlz, pmlz)
    nd = 5
#   z_config = (0.1, 0.8, Dirichlet(), Dirichlet())

    d = RectangularDomain(z_config)

    m = CartesianMesh(d, nd)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 1
    shots = []

    # Define source location and type
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))

    # Define set of receivers
    receiver = PointReceiver(m, (zpos))
    # receivers = ReceiverSet([receiver])

    # Create and store the shot
    shot = Shot(source, receiver)
    # shot = Shot(source, receivers)
    shots.append(shot)

    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver1 = ConstantDensityAcousticWave(m,
                                          formulation='scalar',
                                          spatial_accuracy_order=2,
                                          trange=trange)

    solver2 = ConstantDensityAcousticWave(m,
                                          kernel_implementation='cpp',
                                          formulation='scalar',
                                          spatial_accuracy_order=2,
                                          trange=trange)

    # Generate synthetic Seismic data
    print('Generating data...')
    m_base = solver1.ModelParameters(m, {'C': C})

    # Generate synthetic Seismic data
    sys.stdout.write('Generating data...')
    # m_base=solver.ModelParameters(m, {'C': C})
    g_base = m_base.Perturbation(m)

    m_base.data = np.random.normal(0.0, 1.0, m_base.data.shape)
    m_list = []
    g_list = []

    for i in range(n_mem+1):
        m_tmp = copy.deepcopy(m_base)
        m_tmp.data = np.random.normal(0.0, 1.0, m_base.data.shape)
        m_list.append(m_tmp)
        g_tmp = copy.deepcopy(g_base)
        g_tmp.data = np.random.normal(0.0, 1.0, m_base.data.shape)
        g_list.append(g_tmp)

    memory = deque([], n_mem)

    for i in range(n_mem):
        s_k = m_list[i+1]-m_list[i]
        y_k = g_list[i+1]-g_list[i]
        s_k.data = np.ones(s_k.data.shape)*(i+2)        
        y_k.data = np.ones(y_k.data.shape)
        # y_k.data[1] = 2.0
        rho_k = 1./y_k.inner_product(s_k)
        memory.append([rho_k, s_k, y_k])

    H1 = LBFGS_Hessian(memory)
    x  = copy.deepcopy(g_tmp)
    x.data = np.ones(m_base.data.shape)
    c = H1.inv(x)
    b = H1 * c
    print(b)

    H2 = np.zeros((nd,nd))
    H3 = np.zeros((nd,nd))
    for i in range(nd):
        x_tmp = copy.deepcopy(g_tmp)
        x_tmp.data = np.zeros(m_base.data.shape)
        x_tmp.data[i] = 1.0
        b_tmp = H1 * x_tmp
        c_tmp = H1.inv(x_tmp)
        H2[:, i] = b_tmp.data.flatten()
        H3[:, i] = c_tmp.data.flatten()


    lowerbound = -5.0
    upperbound = -3.0

    x_0 = copy.deepcopy(g_tmp)
    x_0.data = np.ones(m_base.data.shape) *(-4.0)

    proj_op=BoundProjection(lowerbound, upperbound)
    # proj_op = LineProjection()
    grad = copy.deepcopy(g_tmp)
    grad.data = np.ones(m_base.data.shape)

    x_ref = copy.deepcopy(g_tmp)
    x_ref.data = np.zeros(m_base.data.shape)
    objective_fun = Quadratic_obj(grad, H1, x_ref)
    opt_solver = ProjectedGradientDescent(objective_fun, proj_op)

    x_out = opt_solver(200, 100, x_0, verbose=True)











    a = 1

    # test_H = LBFGS_Hessian()
    # b = np.mat([0,1])
    # b = b.transpose()
    # c = test_H.inv(b)
    # d = test_H * b
    # e = b*test_H
    # print('c=', c)
    # print('d=', d)




