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
from pysit.vis.vis import *
from pysit.util.compute_tools import *

from pysit.util.parallel import *

if __name__ == '__main__':
    # Set up domain, mesh and velocity model
    pmlx = PML(0.1, 100)
    pmly = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    y_config = (0.1, 0.9, pmly, pmly)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, y_config, z_config)

    m = CartesianMesh(d, 46, 41, 36)
    
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    Nshots = 1,1

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
    trange = (0.0,3.0)

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

    n_data = (46, 41, 36)
    n_dataplt = (n_data[0], n_data[2], n_data[1])
    Cplot = np.reshape(C, n_data)
    Cplot = np.transpose(Cplot, (0, 2, 1))
    origins = [0.1, 0.1, 0.1]
    deltas = [0.02, 0.02, 0.02]
    
    axis_ticks = [np.array(list(range(0, n_dataplt[0]-5, (n_data[0]-6)//4))), 
                  np.array(list(range(5, n_dataplt[1]-5, (n_data[1]-11)//4))),
                  np.array(list(range(0, n_dataplt[2], (n_data[2]-1)//2)))
                 ]
    axis_tickslabels = [(axis_ticks[0] * deltas[0] * 1000.0 + origins[0] * 1000.0).astype(int),
                        (axis_ticks[1] * deltas[1] * 1000.0 + origins[1] * 1000.0).astype(int),
                        (axis_ticks[2] * deltas[2] * 1000.0 + origins[2] * 1000.0).astype(int)
                       ]

    
    plot_3D_panel(Cplot, slice3d=(22, 18, 20),
                  axis_label=['x [m]', 'z [m]', 'y [m]'],
                  axis_ticks=axis_ticks,
                  axis_tickslabels=axis_tickslabels,
                 )
    plt.title('Slice at \n x = 540 m, y = 500 m, z = 440 m')
    plt.show()

    data = shots[0].receivers.data
    
    t_smp = np.linspace(trange[0], trange[1], data.shape[0])

    fig=plt.figure()
    n_recdata = [len(t_smp), n_data[0], n_data[1]]
    n_recdataplt = [n_data[0], len(t_smp), n_data[1]]
    data = np.reshape(data, n_recdata)
    dataplt = np.transpose(data, (1, 0, 2))
    deltas_data = [deltas[0], solver.dt,  deltas[2]]
    origins_data = [origins[0], 0.0,origins[2]]
    
    axis_ticks = [np.array(list(range(0, n_recdataplt[0]-5, (n_recdataplt[0]-1)//4))), 
                  np.array(list(range(0, n_recdataplt[1]-5, (n_recdataplt[1]-1)//4))),
                  np.array(list(range(0, n_recdataplt[2], (n_recdataplt[2]-1)//2)))
                 ]
    axis_tickslabels = [np.round(axis_ticks[0] * deltas_data[0]  + origins_data[0], 2),
                        np.round(axis_ticks[1] * deltas_data[1]  + origins_data[1], 2),
                        np.round(axis_ticks[2] * deltas_data[2]  + origins_data[2], 2)
                       ]

    
    plot_3D_panel(dataplt, slice3d=(22, 900, 20),
                  axis_label=[ 'x [km]', 'Time [s]', 'y [km]'],
                  axis_ticks=axis_ticks,
                  axis_tickslabels=axis_tickslabels,
                  width_ratios=[1,1], height_ratios=[1,1],cmap='seismic', vmin=-0.2,vmax=0.2
                 )
    plt.show()


    n_timesmp = shots[0].receivers.data.shape[0]
    T_max = solver.tf
    freq_band =[1.0, 30.0]
    filter_op1 = band_pass_filter(n_timesmp, T_max, freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)
    
    objective = TemporalLeastSquares(solver, filter_op=filter_op1, imaging_period=1)

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

    ## Check result

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    
    Cplot = np.reshape(C, n_data)
    Cplot = np.transpose(Cplot, (0, 2, 1))
    origins = [0.1, 0.1, 0.1]
    deltas = [0.02, 0.02, 0.02]
    
    axis_ticks = [np.array(list(range(0, n_dataplt[0]-5, (n_data[0]-6)//4))), 
                  np.array(list(range(5, n_dataplt[1]-5, (n_data[1]-11)//4))),
                  np.array(list(range(0, n_dataplt[2], (n_data[2]-1)//2)))
                 ]
    axis_tickslabels = [(axis_ticks[0] * deltas[0] * 1000.0 + origins[0] * 1000.0).astype(int),
                        (axis_ticks[1] * deltas[1] * 1000.0 + origins[1] * 1000.0).astype(int),
                        (axis_ticks[2] * deltas[2] * 1000.0 + origins[2] * 1000.0).astype(int)
                       ]

    
    plot_3D_panel(Cplot, slice3d=(22, 18, 20),
                  axis_label=['x [m]', 'z [m]', 'y [m]'],
                  axis_ticks=axis_ticks,
                  axis_tickslabels=axis_tickslabels,
                  vmin=1.6, vmax=2.2
                 )
    plt.title('True model Slice at \n x = 540 m, y = 500 m, z = 440 m')
    
    Cout = result.C
    
    Coutplot = np.reshape(Cout, n_data)
    Coutplot = np.transpose(Coutplot, (0, 2, 1))
    origins = [0.1, 0.1, 0.1]
    deltas = [0.02, 0.02, 0.02]
    
    axis_ticks = [np.array(list(range(0, n_dataplt[0]-5, (n_data[0]-6)//4))), 
                  np.array(list(range(5, n_dataplt[1]-5, (n_data[1]-11)//4))),
                  np.array(list(range(0, n_dataplt[2], (n_data[2]-1)//2)))
                 ]
    axis_tickslabels = [(axis_ticks[0] * deltas[0] * 1000.0 + origins[0] * 1000.0).astype(int),
                        (axis_ticks[1] * deltas[1] * 1000.0 + origins[1] * 1000.0).astype(int),
                        (axis_ticks[2] * deltas[2] * 1000.0 + origins[2] * 1000.0).astype(int)
                       ]

    
    plot_3D_panel(Coutplot, slice3d=(22, 18, 20),
                  axis_label=['x [m]', 'z [m]', 'y [m]'],
                  axis_ticks=axis_ticks,
                  axis_tickslabels=axis_tickslabels,
                  vmin=1.6, vmax=2.2
                 )
    plt.title('Reconstructed model Slice at \n x = 540 m, y = 500 m, z = 440 m')
    plt.show()
    
    



