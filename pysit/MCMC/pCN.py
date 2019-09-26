


import sys
import time
import copy
import tensorflow as tf

import numpy as np
import scipy.io as sio
from pysit.util.io import *
from pysit.util.parallel import ParallelWrapShotNull

__all__=['pCN']

__docformat__ = "restructuredtext en"

class pCN(object):
    """ Class of pCN type MCMC method for UQ and optimization for CNN type variables
    ----------
    solver : pysit wave solver object
        A wave solver that inherits from pysit.solvers.WaveSolverBase
    ivnersion_methods : class
        A class containing all of the methods required to compute the inversion steps.
    verbose : bool
        Verbosity flag.
    xi : solver.WaveSolverParameters
        Current state of the unknowns.
    i : float
        Current iteration index.
    <blank>_history : list of tuples
        History of property (e.g., step length) with entries like the tuple (i, step_length_i) for index i.
    <blank>_frequency : int
        Iteration frequency at which to store a particular property.

    """

    def __init__(self, objective):
        """Constructor for the BasicDescentAlgorithm class.

        Parameters
        ----------
        solver : pysit wave solver object
            A wave solver that inherits from pysit.solvers.WaveSolverBase
        inversion_methods : class
            A class containing all of the methods required to compute the inversion steps.

        Notes
        -----
        * InversionMethodsType is a data type that takes a wave solver object
          as a construction argument.  The collection of inversion methods will
          depend on the solver.
        * InversionMethodsType must have member functions that implement the
          basic wave imaging procedures, e.g., forward modeling,
          adjoint modeling, demigration, etc.

        """
        self.objective_function = objective
        self.solver = objective.solver
        self.verbose = False

        self.use_parallel = objective.use_parallel()

        self.max_linesearch_iterations = 10

        self.logfile = sys.stdout
        self.proj_op = None

        self.write = False

    def __call__(self,
                 shots,
                 initial_model,
                 n_cnn_para,
                 nsmps,
                 beta,
                 noise_sigma=1.0,
                 isuq=False,
                 print_interval=10,
                 save_interval=None,
                 initial_value_cnn=None,
                 parallel_wrap=ParallelWrapShotNull(),
                 verbose=False,
                 append=False,
                 write=False,
                 **kwargs):
        """The main function for executing a number of steps of the descent
        algorith.

        Most things can be done without directly overriding this function.

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute on.
        initial_value : solver.WaveParameters
            Initial guess for the iteration.
        iteration_parameters : int, iterable
            Loop iteration parameters, like number of steps or frequency sets.
        <blank>_frequency : int, optional kwarg
            Frequency with which to store histories.  Detailed in reset method.
        verbose : bool
            Verbosity flag.
        linesearch_configuration : dictionary
            Possible parameters for linesearch, for more details, please check the introduction of the function set_linesearch_configuration

        """
        if initial_value_cnn is None:
            m0_cnn = tf.random.uniform([1, n_cnn_para])
        else:
            m0_cnn = initial_value_cnn

        phi0 = self.objective_function.evaluate(shots, initial_model, m0_cnn) / noise_sigma**2.0
        Ms = []
        A_accept = []
        Phi = []
        Beta = []
        Ms.append(m0_cnn)
        if parallel_wrap.use_parallel:
            if parallel_wrap.comm.Get_rank() == 0:
                r_probs = np.random.uniform(0.0, 1.0, nsmps)
            else:
                r_probs = None

            r_probs = parallel_wrap.comm.bcast(r_probs, root=0) 
                
        else:
            r_probs = np.random.uniform(0.0, 1.0, nsmps)

        m_min_cnn = m0_cnn
        phi_min = phi0
        

        for i in range(nsmps):
            # mtmp_cnn = tf.random.uniform([1, n_cnn_para])
            Beta.append(beta)
            if parallel_wrap.use_parallel:
                if parallel_wrap.comm.Get_rank() == 0:
                    mtmp_cnn = tf.random.normal([1, n_cnn_para])
                else:
                    mtmp_cnn = None

                mtmp_cnn = parallel_wrap.comm.bcast(mtmp_cnn, root=0) 
                
            else:
                mtmp_cnn = tf.random.normal([1, n_cnn_para])

            m1_cnn = np.sqrt(1-beta**2.0)*m0_cnn + beta*mtmp_cnn
            phi1 = self.objective_function.evaluate(shots, initial_model, m1_cnn) / noise_sigma**2.0

            if phi1 < phi_min:
                phi_min = phi1
                m_min_cnn = m1_cnn

            a_accept = np.min((np.exp(phi0-phi1), 1))
            A_accept.append(a_accept)
            Phi.append(phi1)
            # print('Accept probability:', a_accept)
            
            if np.mod(i,print_interval) == 0:
                if self.use_parallel is True:
                    if self.objective_function.parallel_wrap_shot.comm.Get_rank() == 0:
                        print('Iteration:', i)
                        print('f: ', phi_min)
                else:
                    print('Iteration:', i)
                    print('f: ', phi_min)

            if a_accept > r_probs[i]:
                Ms.append(m1_cnn)
                m0_cnn = m1_cnn
                phi0 = phi1
                beta *= 1.2
            else:
                Ms.append(m0_cnn)
                if a_accept < 0.1:
                    if beta > 1e-4:
                        beta *= 0.5

            if save_interval is not None:
                if np.mod(i,save_interval) == 0:
                    if (parallel_wrap.use_parallel is None) or (parallel_wrap.comm.Get_rank() == 0):
                        if i == 0:
                            Snp = np.array(Ms)
                        else:
                            Msi = Ms[len(Snp):len(Ms)]
                            Snpi = np.array(Msi)
                            nsize = np.shape(Snpi)
                            Snpi = np.reshape(Snpi, [nsize[0], nsize[-1]])
                            print(np.shape(Snp))
                            print(np.shape(Snpi))
                            Snp = np.concatenate((Snp, Snpi), axis=0)
                        
                        n_size = np.shape(Snp)
                        n_size_new = [n_size[0], n_size[-1]]
                        Snp = np.reshape(Snp, n_size_new)
                        write_data('./Samples.mat', Snp, [1,1], [1,1], n_size_new)
                        write_data('./MAP.mat', np.array(m_min_cnn), [1,1], [1,1], np.array(m_min_cnn).shape)
                        write_data('./probability.mat', A_accept, [1], [1], len(A_accept))
                        write_data('./objective_function.mat', Phi, [1], [1], len(Phi))
                        write_data('./betas.mat', Beta, [1], [1], len(Beta))

                    if parallel_wrap.use_parallel is not None:
                        parallel_wrap.comm.Barrier()


            

        result = dict()
        result['MAP'] = m_min_cnn
        result['samples'] = Ms
        result['accept_prob'] = A_accept
        result['Phi'] = Phi

        return result

            


            






