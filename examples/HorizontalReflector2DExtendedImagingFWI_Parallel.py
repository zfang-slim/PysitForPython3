# Std import block
import time
import os

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from mpi4py import MPI

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.gallery import three_layered_medium
from pysit.util.parallel import *


if __name__ == '__main__':

    # Setup
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    os.environ["OMP_NUM_THREADS"] = "4"

    pwrap = ParallelWrapShot()

    nsteps = 10

    # modeling.frequency_modeling.extended_modeling_test()

    C, C0, m, d = three_layered_medium(initial_model_style='smooth',
                                       initial_config={'sigma': 20.5, 'filtersize': 4},
                                       vels=(1.5, 2.0, 2.5),
                                       pml_width=[0.1, 0.1],
                                       nx=31, nz=31,
                                       water_layer_depth=0.01)

    C, C00, m, d = three_layered_medium(initial_model_style='smooth',
                                       initial_config={'sigma': 2.5, 'filtersize': 4},
                                       vels=(1.5, 2.0, 2.5),
                                       pml_width=[0.1, 0.1],
                                       nx=31, nz=31,
                                       water_layer_depth=0.01)
    
    # C00 = np.ones(C00.shape) * 1.5

    C1 = np.reshape(C0, (31, 31))
    C2 = np.reshape(C00, (31, 31))
    plt.figure()
    plt.imshow(C1)

    plt.figure()
    plt.imshow(C2)

    max_sub_offset = 0.05
    h = 0.01

    d_slow = 1/C**2.0 - 1/C0**2.0

    m1_extend = ExtendedModelingParameter2D(m, max_sub_offset, h)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = 0.04

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=4,
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

    fh = TemporalExtendedImagingInversion(solver_time, h, max_sub_offset=max_sub_offset, krylov_maxiter=20, 
                                          regularization_value=10.0**4.0, weight_matrix='linear_h',
                                          parallel_wrap_shot=pwrap)

    for i in range(len(shots)):
        shots[i].reset_time_series(solver_time.ts())
        shots[i].receivers.data = lindatas[i]
        
    vel_bound = [1.3, 3.0]
    PMLproj = PMLExtensionPrj()
    Boxproj = BoxConstraintPrj(vel_bound)

    # invalg = PQN(fh, memory_length=10, proj_op=Boxproj)
    invalg = PQN(fh, memory_length=10)

    status_configuration = {'value_frequency': 1,
                            'residual_frequency': 1,
                            'residual_length_frequency': 1,
                            'objective_frequency': 1,
                            'step_frequency': 1,
                            'step_length_frequency': 1,
                            'gradient_frequency': 1,
                            'gradient_length_frequency': 1,
                            'run_time_frequency': 1,
                            'alpha_frequency': 1,
                            }

    line_search = 'backtrack'

    initial_value = solver_time.ModelParameters(m, {'C': C00})
    # initial_value.data = np.ones(base_model.data.shape) * 1.5

    fh0 = fh.evaluate(shots, initial_value)

    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True, write=True)

    # g = fh.compute_gradient(shots, m0)

    # fh0 = fh.evaluate(shots, m0)
    # print(fh0)

    # LSM_obj = ExtendLSM(tools, shots, m0, lindatas, max_sub_offset, h,
    #                     imaging_period=1, krylov_maxiter=2, parallel_wrap_shot=pwrap)
    # print(LSM_obj.parallel_wrap_shot.use_parallel)
    # x_out = LSM_obj.run_lsm()

    # # GenerateDataObj.linear_forward_model(shots[0], m0, m1, frequencies, ['simdata'])
    # print('Linearized data is generated')

    # A = np.reshape(x_out.data, m1_extend.full_sh_data)
    # plt.figure()
    # plt.imshow(A[25, :, :], interpolation='nearest', aspect='auto')

    # import pickle
    # file_pi = open('ELSM_Result_test.obj', 'wb')
    # pickle.dump(x_out, file_pi)
    # file_pi.close()

    # file_pi = open('ELSM_TrueModel_test.obj', 'wb')
    # pickle.dump(m1_extend, file_pi)
    # file_pi.close()

    # import pickle
    # filehandler = open(filename, 'rb')
    # object = pickle.load(filehandler)

    a = 1
