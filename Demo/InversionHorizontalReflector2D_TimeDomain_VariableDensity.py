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

    solver = VariableDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         max_C=2.5)
    
    solver.max_C = 2.5
    
    base_model = solver.ModelParameters(m,model_param)
    
    generate_seismic_data(shots, solver, base_model)
    

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

    # Set up the inversion

    n_timesmp = shots[0].receivers.data.shape[0]
    T_max = solver.tf
    freq_band =[1.0, 30.0]
    filter_op1 = band_pass_filter(n_timesmp, T_max, freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)
    
    # Least-squares objective function
    objective = TemporalLeastSquares(solver, filter_op=filter_op1, imaging_period=1)

    # Envelope objective function
    # objective = TemporalEnvelope(solver, envelope_power=2.0, filter_op=filter_op1, imaging_period=1)

    # Cross-correlation objective function
    # objective = TemporalCorrelate(solver, filter_op=filter_op, imaging_period=1)    

    # Optimal transportation objective function with linear transformation
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='linear', c_ratio=2.0)

    # Optimal transportation objective function with quadratic transformation
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='quadratic')

    # Optimal transportation objective function with absolute transformation
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='absolute')

    # Optimal transportation objective function with exponential transformation
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='exponential', exp_a=1.0)    



    bound = [1.5, 3.0]
    Proj_Op1 = BoxConstraintPrj(bound)
    invalg = PQN(objective, proj_op=None, memory_length=10)
    
    nsteps = 5
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
    
    result = invalg(shots, initial_value, nsteps,
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