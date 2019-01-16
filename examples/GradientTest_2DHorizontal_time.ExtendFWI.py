# Std import block
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from pysit import *
from pysit.gallery import horizontal_reflector

from GradientTest import GradientTest
from pysit.util.parallel import *
from pysit.gallery import three_layered_medium

if __name__ == '__main__':
    # Setup

    # Setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    os.environ["OMP_NUM_THREADS"] = "4"

    pwrap = ParallelWrapShot()

    nsteps = 5

    # modeling.frequency_modeling.extended_modeling_test()

    C, C0, m, d = three_layered_medium(initial_model_style='smooth',
                                       initial_config={'sigma': 20.5, 'filtersize': 4},
                                       vels=(1.5, 2.0, 2.5),
                                       pml_width=[0.1, 0.1],
                                       nx=31, nz=31,
                                       water_layer_depth=0.01)

    C, C00, m, d = three_layered_medium(initial_model_style='smooth',
                                        initial_config={'sigma': 10.5, 'filtersize': 4},
                                        vels=(1.5, 2.0, 2.5),
                                        pml_width=[0.1, 0.1],
                                        nx=31, nz=31,
                                        water_layer_depth=0.01)

    # C1 = np.reshape(C, (121, 61))
    # C2 = np.reshape(C0, (121, 61))
    # plt.figure()
    # plt.imshow(C1)

    # plt.figure()
    # plt.imshow(C2)

    max_sub_offset = 0.01
    h = 0.01

    d_slow = 1/C**2.0 - 1/C0**2.0

    m1_extend = ExtendedModelingParameter2D(m, max_sub_offset, h)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = 0.04

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   parallel_shot_wrap=pwrap,
                                   )

    # Define and configure the wave solver
    trange = (0.0, 0.5)

    # solver = ConstantDensityHelmholtz(m,
    #                                   spatial_accuracy_order=4)

    solver_time = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=4,
                                              kernel_implementation='omp',
                                              trange=trange,
                                              max_C=3.0)

    solver_time.max_C = 3.0

    # base_model = solver_time.ModelParameters(m, {'C': C})
    # generate_seismic_data(shots, solver_time, base_model)
    # shots_t = copy.deepcopy(shots)
    base_model = solver_time.ModelParameters(m, {'C': C0})
    # generate_seismic_data(shots_t, solver_time, base_model)
    # plt.figure()
    # plt.imshow(shots[0].receivers.data-shots_t[0].receivers.data, interpolation='nearest', aspect='auto')

    m0 = base_model

    tools = TemporalModeling(solver_time)
    m1_extend.setter(np.zeros(m1_extend.sh_data))
    d_m = d_slow
    dmtmp = d_m.data
    m1 = m0.perturbation()
    sh_true = m1.mesh._shapes[(False, True)]
    dmtmp = np.reshape(dmtmp, sh_true)
    sh_cut = m1_extend.sh_sub
    dmtmp = dmtmp[0:sh_cut[0], :]
    # dmtmp = np.zeros(dmtmp.shape)
    # dmtmp[:, 40] = 1.0
    dmtmp = dmtmp.reshape(-1)
    m1_extend.data[:, (m1_extend.sh_data[1]-1)//2] = dmtmp

    m1 = m0.perturbation()
#   m1 += M
    # dmtmp = d_m.data
    # dmtmp[np.where(dmtmp <= 2)] = 0

    linfwdret = tools.linear_forward_model_extend(shots, m0, m1_extend,
                                                  max_sub_offset, h, ['simdata'])
    lindatas = linfwdret['simdata']

    m2_extend = copy.deepcopy(m1_extend)
    m2_extend.data = m2_extend.data * 0.9
    m2_extend.data = np.zeros(m2_extend.data.shape)
    m2_extend.data[:, (m1_extend.sh_data[1]-1)//2 + 1] = dmtmp
    m2_extend.data[:, (m1_extend.sh_data[1]-1)//2 - 1] = dmtmp
    m2_extend.data = np.random.normal(0,1,m2_extend.data.shape)

    objective = TemporalExtendedImagingInversionSub(solver_time, h, max_sub_offset=max_sub_offset, krylov_maxiter=20, parallel_wrap_shot=pwrap, dm_extend=m2_extend)
    solver = solver_time

    for i in range(len(shots)):
        shots[i].reset_time_series(solver_time.ts())
        shots[i].receivers.data = lindatas[i]



    # Define the inversion algorithm
    grad_test = GradientTest(objective)
    grad_test.base_model = solver.ModelParameters(m, {'C': C00})
    grad_test.length_ratio = np.power(5.0, range(-8, -0))

    # Set up the perturbation direction
    dC_vec = copy.deepcopy(grad_test.base_model)
    m_size = m._shapes[(False, True)]
    tmp = np.random.normal(0, 1, m_size)
    # tmp = np.ones(m_size)
    tmp[0:3, :] = 0.0
    tmp[m_size[0]-3:m_size[0], :] = 0.0
    tmp[:, 0:3] = 0.0
    tmp[:, m_size[1]-3:m_size[1]] = 0.0
    tmp = np.reshape(tmp, grad_test.base_model.data.shape)
    dC_vec.data = tmp
    norm_dC_vec = np.linalg.norm(dC_vec.data)
    norm_base_model = np.linalg.norm(grad_test.base_model.data)
    dC_vec.data = dC_vec.data * 0.1 * (norm_base_model / norm_dC_vec)
    grad_test.model_perturbation = dC_vec
    # Execute inversion algorithm
    print('Gradient test ...')
    tt = time.time()

    result = grad_test(shots)

    print('...run time:  {0}s'.format(time.time()-tt))

    print(grad_test.objective_value)

    plt.figure()
    plt.loglog(grad_test.length_ratio, grad_test.zero_order_difference, 'b',
               grad_test.length_ratio, grad_test.length_ratio, 'r')
    plt.title('Zero order difference')
    plt.gca().legend(('df_0', 'h'))

    plt.figure()
    plt.loglog(grad_test.length_ratio, grad_test.first_order_difference, 'b',
               grad_test.length_ratio, np.power(grad_test.length_ratio, 1.0), 'y',
               grad_test.length_ratio, np.power(grad_test.length_ratio, 2.0), 'r')
    plt.title('First order difference')
    plt.gca().legend(('df_1', 'h', 'h^2'))

    plt.show()

