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
                                   receiver_kwargs={}
                                   )
    
    shots_freq = copy.deepcopy(shots)

    # Define and configure the wave solver
    trange = (0.0,2.0)

    # Define the time-domain wave-equation solver and generate the time-domain data

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp',
                                         max_C=2.5)

    solver.max_C = 2.5
    
    base_model = solver.ModelParameters(m,{'C': C})
    
    generate_seismic_data(shots, solver, base_model)

    # Check the result and plot the result

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
    invalg = PQN(objective, proj_op=Proj_Op1, memory_length=10)
    
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
    
    initial_value = solver.ModelParameters(m,{'C': C0})
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
    plt.figure(figsize=(12,16))
    plt.subplot(3,1,1)
    vis.plot(C0, m, clim=clim)
    plt.title('Initial Model')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,1,2)
    vis.plot(C, m, clim=clim)
    plt.title('True Model')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()
    plt.subplot(3,1,3)
    vis.plot(result.C, m, clim=clim)
    plt.title('Reconstruction')
    plt.xlabel('X [km]')
    plt.ylabel('Z [km]')
    plt.colorbar()

    plt.show()