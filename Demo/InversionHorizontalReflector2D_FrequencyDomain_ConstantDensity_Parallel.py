import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from shutil import copy2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpi4py import MPI

import sys
import scipy.io as sio

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.io import *
from pysit.util.compute_tools import *

from pysit.util.parallel import *

if __name__ == '__main__':
    # Set up parallel computing environment
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    # Set up domain, mesh and velocity model
    pmlx = PML(0.1, 1000, compact=True)
    pmlz = PML(0.1, 1000, compact=True)

    x_config = (0.0, 2.0, pmlx, pmlx)
    z_config = (0.0, 1.0, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, 201, 101)

    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./10.)*zmax

    Nshots = 3

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=Nshots,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   parallel_shot_wrap=pwrap,
                                   )
    
    shots_freq = copy.deepcopy(shots)

    # Define and configure the wave solver
    trange = (0.0,2.0)

    # Define the time-domain wave-equation solver and generate the time-domain data

    solver =ConstantDensityHelmholtz(m, spatial_accuracy_order=4)

    
    base_model = solver.ModelParameters(m,{'C': C})
    
    frequencies = [2.0,3.0]
    
    print('Generating data...')
    tt = time.time()
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies)
    print('Data generation: {0}s'.format(time.time()-tt))

    # Check the result and plot the result

    if rank == 0:

        clim = C.min(),C.max()
        plt.figure(figsize=(20,4))
        plt.subplot(1,2,1)
        vis.plot(C0, m, clim=clim)
        plt.title(r'Initial Model of $v$')
        plt.colorbar()
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.subplot(1,2,2)
        vis.plot(C, m, clim=clim)
        plt.title(r"True Model of $v$")
        plt.colorbar()
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.title('Imaginary part of data at f = 3.0Hz')
        plt.show()

    comm.Barrier()

    data = shots[0].receivers.data_dft
    
    
    xrec = np.linspace(0.0,2.0,201)
    data1 = shots[0].receivers.data_dft[2.0]
    data2 = shots[0].receivers.data_dft[3.0]
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.plot(xrec, np.real(data1.flatten()))
    plt.xlabel('Receivers [km]')
    plt.title('Real part of data at f = 2.0Hz')
    plt.subplot(2,2,2)
    plt.plot(xrec, np.real(data2.flatten()))
    plt.xlabel('Receivers [km]')
    plt.title('Real part of data at f = 3.0Hz')

    plt.subplot(2,2,3)
    plt.plot(xrec, np.imag(data1.flatten()))
    plt.xlabel('Receivers [km]')
    plt.title('Imaginary part of data at f = 2.0Hz')
    plt.subplot(2,2,4)
    plt.plot(xrec, np.imag(data2.flatten()))
    plt.xlabel('Receivers [km]')
    plt.title('Imaginary part of data at f = 3.0Hz')
    plt.show()



    # Set up the inversion

    objective = FrequencyLeastSquares(solver, parallel_wrap_shot=pwrap)


    bound = [1.5, 3.0]
    Proj_Op1 = BoxConstraintPrj(bound)
    invalg = PQN(objective, proj_op=Proj_Op1, memory_length=10)
    loop_configuration = [(5, {'frequencies': [2.0]}), (5, {'frequencies': [3.0]})]

    status_configuration = {'value_frequency'           : 1,
                            'residual_frequency'        : 1,
                            'residual_length_frequency' : 1,
                            'objective_frequency'       : 1,
                            'step_frequency'            : 1,
                            'step_length_frequency'     : 1,
                            'gradient_frequency'        : 1,
                            'gradient_length_frequency' : 1,
                            'run_time_frequency'        : 1,
                            'alpha_frequency'           : 1,
                            }
    
    initial_value = solver.ModelParameters(m,{'C': C0})
    line_search = 'backtrack'
    
    result = invalg(shots, initial_value, loop_configuration,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)

    if rank == 0:

    # Check result

        obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

        plt.figure()
        plt.semilogy(obj_vals)
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.show()
        
        clim = C.min(),C.max()

        # Do something to visualize the results
        plt.figure(figsize=(16,10))
        plt.subplot(2,2,1)
        vis.plot(C0, m, clim=clim)
        plt.title('Initial Model')
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.colorbar()
        plt.subplot(2,2,2)
        vis.plot(C, m, clim=clim)
        plt.title('True Model')
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.colorbar()
        plt.subplot(2,2,3)
        vis.plot(result.C, m, clim=clim)
        plt.title('Reconstruction')
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.colorbar()
        plt.subplot(2,2,4)
        vis.plot(result.C-C0, m, clim=[-.01,.01])
        plt.title('Difference')
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        plt.colorbar()

        plt.show()

    comm.Barrier()