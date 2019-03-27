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
from pysit.vis.vis import *

from pysit.util.parallel import *

if __name__ == '__main__':
    # Set up parallel computing environment
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

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

    Nshots = 1,2

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
    trange = (0.0,3.0)

    # Define the time-domain wave-equation solver and generate the time-domain data

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')
    
    base_model = solver.ModelParameters(m,{'C': C})

    print('Generating time-domain data...')
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    print('Time-domain data generation: {0}s'.format(time.time()-tt))
    

    # Check the result and plot the result
    n_data = (46, 41, 36)
    n_dataplt = (n_data[0], n_data[2], n_data[1])
    origins = [0.1, 0.1, 0.1]
    deltas = [0.02, 0.02, 0.02] 

    if rank == 0:  
      Cplot = np.reshape(C, n_data)
      Cplot = np.transpose(Cplot, (0, 2, 1))
       
      
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

    comm.Barrier()

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


    # Define the frequency-domain wave-equation solver and generate the frequency-domain data

    pmlx = PML(0.1, 100, compact=True)
    pmly = PML(0.1, 100, compact=True)
    pmlz = PML(0.1, 100, compact=True)

    x_config = (0.1, 1.0, pmlx, pmlx)
    y_config = (0.1, 0.9, pmly, pmly)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, y_config, z_config)

    m = CartesianMesh(d, 46, 41, 36)
    
    C, C0, m, d = horizontal_reflector(m)
    
    solver = ConstantDensityHelmholtz(m,
                                      spatial_accuracy_order=4,
                                      parallel_shot_wrap=pwrap,)
    
    frequencies = [2.0,3.0]
    
    print('Generating frequency-domain data...')
    tt = time.time()
    generate_seismic_data(shots_freq, solver, base_model, frequencies=frequencies, petsc='mumps')
    print('Frequency-domain data generation: {0}s'.format(time.time()-tt))

    # Check the result and plot the result
    xrec = np.linspace(0.1,1.0,46)
    yrec = np.linspace(0.1,0.9,41)
    data1 = shots_freq[0].receivers.data_dft[2.0]
    data2 = shots_freq[0].receivers.data_dft[3.0]
    data1 = np.reshape(data1, (len(xrec),len(yrec)))
    data2 = np.reshape(data2, (len(xrec),len(yrec)))
    
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    vmax = np.abs(np.real(data1)).max()
    clim=np.array([-vmax, vmax])
    plt.imshow(np.real(data1).transpose(),cmap='seismic',clim=clim,
              extent=[xrec[0], xrec[-1], yrec[-1], yrec[0]])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Real part of data at 2 Hz')
    plt.colorbar()
    
    plt.subplot(2,2,2)
    vmax = np.abs(np.imag(data1)).max()
    clim=np.array([-vmax, vmax])
    plt.imshow(np.imag(data1).transpose(),cmap='seismic',clim=clim,
              extent=[xrec[0], xrec[-1], yrec[-1], yrec[0]])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Imaginary part of data at 2 Hz')
    plt.colorbar()
    
    plt.subplot(2,2,3)
    vmax = np.abs(np.real(data2)).max()
    clim=np.array([-vmax, vmax])
    plt.imshow(np.real(data2).transpose(),cmap='seismic',clim=clim,
              extent=[xrec[0], xrec[-1], yrec[-1], yrec[0]])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Real part of data at 3 Hz')
    plt.colorbar()
    
    plt.subplot(2,2,4)
    vmax = np.abs(np.imag(data2)).max()
    clim=np.array([-vmax, vmax])
    plt.imshow(np.imag(data2).transpose(),cmap='seismic',clim=clim,
              extent=[xrec[0], xrec[-1], yrec[-1], yrec[0]])
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Imaginary part of data at 3 Hz')
    plt.colorbar()
    plt.show()
    





