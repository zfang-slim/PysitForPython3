# Std import block
import time
import os

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import copy

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':

    # modeling.frequency_modeling.extended_modeling_test()

    # Setup
    hybrid = False
    # enable Open MP multithread solver
    os.environ["OMP_NUM_THREADS"] = "4"

    #   Define Domain
    pmlx = PML(0.1, 300)
    pmlz = PML(0.1, 300)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 91, 71)

    max_sub_offset = 0.2
    h = 0.01

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    m1_extend = ExtendedModelingParameter2D(m, max_sub_offset, h)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=3,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    trange = (0.0, 2.5)

    solver = ConstantDensityHelmholtz(m,
                                      spatial_accuracy_order=4)

    solver_time = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=4,
                                              kernel_implementation='omp',
                                              trange=trange)

    base_model = solver.ModelParameters(m, {'C': C0})
    d_m = solver.ModelParameters(m, {'C': C})
    frequencies = [2.0, 3.5, 5.0]

    GenerateDataObj = FrequencyModeling(solver)

    # generate_shot_data_frequency(shots[0], solver, base_model, frequencies)

    # GenerateDataObj.forward_model(shots[0], base_model, frequencies, ['simdata'])
    # print('Forward data is generated')

    # shots_fwd = copy.deepcopy(shots)

    m0 = base_model

    tools = TemporalModeling(solver_time)
    m1_extend.setter(np.zeros(m1_extend.sh_data))
    d_m = solver.ModelParameters(m, {'C': C})
    dmtmp = d_m.data
    dmtmp[np.where(dmtmp <= 2)] = 0
    m1 = m0.perturbation()
    sh_true = m1.mesh._shapes[(False, True)]
    dmtmp = np.reshape(dmtmp, sh_true)
    sh_cut = m1_extend.sh_sub
    dmtmp = dmtmp[0:sh_cut[0], :]
    dmtmp = np.zeros(dmtmp.shape)
    dmtmp[:, 40] = 1.0
    dmtmp = dmtmp.reshape(-1)
    m1_extend.data[:, (m1_extend.sh_data[1]-1)//2] = dmtmp

    m1 = m0.perturbation()
#   m1 += M
    # dmtmp = d_m.data
    # dmtmp[np.where(dmtmp <= 2)] = 0

    linfwdret = tools.linear_forward_model_extend(shots, m0, m1_extend,
                                                  max_sub_offset, h, ['simdata'])
    lindatas = linfwdret['simdata']

    LSM_obj = ExtendLSM(tools, shots, m0, lindatas, max_sub_offset, h,
                        imaging_period=1, krylov_maxiter=20)
    x_out = LSM_obj.run_lsm()

    # GenerateDataObj.linear_forward_model(shots[0], m0, m1, frequencies, ['simdata'])
    print('Linearized data is generated')

    A = np.reshape(x_out.data,(49,71,41))
    plt.figure()
    plt.imshow(A[25, :, :], interpolation='nearest', aspect='auto')

    import pickle
    file_pi = open('ELSM_Result.obj', 'wb')
    pickle.dump(x_out, file_pi)
    file_pi.close()

    file_pi = open('ELSM_TrueModel.obj', 'wb')
    pickle.dump(m1_extend, file_pi)
    file_pi.close()

    # import pickle
    # filehandler = open(filename, 'rb')
    # object = pickle.load(filehandler)



    a = 1
