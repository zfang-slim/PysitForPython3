import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from shutil import copy2
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import scipy.io as sio

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.io import *
from pysit.util.compute_tools import *

from pysit.util.parallel import *

if __name__ == '__main__':
    # Set up domain, mesh and velocity model
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

    Nshots = 3

    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=Nshots,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={}
                                   )
    
    shots_freq = copy.deepcopy(shots)

    # Define and configure the wave solver
    trange = (0.0,2.0)

    # Define the time-domain wave-equation solver and generate the time-domain data

    solver = VariableDensityHelmholtz(m,
                                      spatial_accuracy_order=4,
                                      trange=trange,
                                      )
    
    
    base_model = solver.ModelParameters(m,model_param)

    frequencies = [2.0,3.0]
    
    generate_seismic_data(shots, solver, base_model, frequencies=frequencies)
    

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

    objective = FrequencyLeastSquares(solver)

    invalg = PQN(objective, proj_op=None, memory_length=10)
    nsteps = 5
    loop_configuration = [(nsteps, {'frequencies': [2.0]}), (nsteps, {'frequencies': [3.0]})]
    

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
    
    initial_value = solver.ModelParameters(m,model_init)
    line_search = 'backtrack'
    
    result = invalg(shots, initial_value, loop_configuration,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True)
    

    # Check result

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.show()
    
    clim = C.min(),C.max()

    # Do something to visualize the results
    plt.figure(figsize=(20,16))
    clim = rho.min(),rho.max()
    plt.subplot(3,2,1)
    vis.plot(rho0, m, clim=clim)
    plt.title(r'Initial Model of $\rho$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,2,3)
    vis.plot(rho, m, clim=clim)
    plt.title(r'True Model of $\rho$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,2,5)
    vis.plot(result.rho, m, clim=clim)
    plt.title(r'Reconstruction or $\rho$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    
    clim = kappa.min(),kappa.max()
    plt.subplot(3,2,2)
    vis.plot(kappa0, m, clim=clim)
    plt.title(r'Initial Model of $\kappa$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,2,4)
    vis.plot(kappa, m, clim=clim)
    plt.title(r'True Model of $\kappa$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,2,6)
    vis.plot(result.kappa, m, clim=clim)
    plt.title(r'Reconstruction or $\kappa$')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()

    plt.show()