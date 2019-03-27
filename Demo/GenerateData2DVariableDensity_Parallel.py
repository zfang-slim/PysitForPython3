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

from pysit.util.parallel import *

if __name__ == '__main__':

    # Set up parallel computing environment
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    # Set up domain, mesh and density and bulk mudulus model
    pmlx = PML(0.1, 1000)
    pmlz = PML(0.1, 1000)

    x_config = (0.0, 2.0, pmlx, pmlx)
    z_config = (0.0, 1.0, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    nx = 201
    nz = 101

    m = CartesianMesh(d, nx, nz)

    C, C0, m, d = horizontal_reflector(m)

    rho0 = np.ones((nx, nz))
    rho = rho0 + np.reshape(C-C0, (nx,nz))
    
    rho0 = rho0.reshape((nx*nz,1))
    rho = rho.reshape((nx*nz,1))
    kappa0 = rho0 * C0**2.0
    kappa = rho * C**2.0
    
    model_param = {'kappa': kappa, 'rho': rho}
    model_init = {'kappa' : kappa0, 'rho': rho0}

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./10.)*zmax

    Nshots = 2

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

    solver = VariableDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         )
    
    base_model = solver.ModelParameters(m,model_param)
    
    print('Generating time-domain data...')
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    print('Time-domain data generation: {0}s'.format(time.time()-tt))


    # Check the result and plot the result

    # Check the result and plot the result
    clim = rho.min(),rho.max()
    plt.figure(figsize=(20,8))
    plt.subplot(2,2,1)
    vis.plot(rho0, m, clim=clim)
    plt.title(r'Initial Model of $\rho$')
    plt.colorbar()
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.subplot(2,2,2)
    vis.plot(rho, m, clim=clim)
    plt.title(r"True Model of $\rho$")
    plt.colorbar()
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    
    clim = kappa.min(),kappa.max()
    plt.subplot(2,2,3)
    vis.plot(kappa0, m, clim=clim)
    plt.title(r'Initial Model of $\kappa$')
    plt.colorbar()
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.subplot(2,2,4)
    vis.plot(kappa, m, clim=clim)
    plt.title(r"True Model of $\kappa$")
    plt.colorbar()
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.show()

    data = shots[0].receivers.data
    
    t_smp = np.linspace(trange[0], trange[1], data.shape[0])

    fig=plt.figure()
    im1=plt.imshow(data, interpolation='nearest', aspect='auto', cmap='seismic', clim =[-.1,.1],
              extent=[0.0, 2.0, t_smp[-1], 0.0])
    plt.xlabel('Receivers [km]')
    plt.ylabel('Time [s]')
    plt.colorbar()
    plt.show()

    # Define the frequency-domain wave-equation solver and generate the frequency-domain data

    solver = VariableDensityHelmholtz(m,
                                      spatial_accuracy_order=4,
                                      parallel_shot_wrap=pwrap)
    
    frequencies = [2.0,3.0]
    
    print('Generating frequency-domain data...')
    tt = time.time()
    generate_seismic_data(shots_freq, solver, base_model, frequencies=frequencies, petsc='mumps')
    print('Frequency-domain data generation: {0}s'.format(time.time()-tt))




    xrec = np.linspace(0.0,2.0,201)
    data1 = shots_freq[0].receivers.data_dft[2.0]
    data2 = shots_freq[0].receivers.data_dft[3.0]
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
    





