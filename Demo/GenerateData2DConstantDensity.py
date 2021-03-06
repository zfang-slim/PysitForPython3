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

    Nshots = 1

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
                                         kernel_implementation='cpp')
    
    base_model = solver.ModelParameters(m,{'C': C})
    
    print('Generating time-domain data...')
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    print('Time-domain data generation: {0}s'.format(time.time()-tt))

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

    # Define the frequency-domain wave-equation solver and generate the frequency-domain data

    solver = ConstantDensityHelmholtz(m,
                                      spatial_accuracy_order=4)
    
    frequencies = [2.0,3.0]
    
    print('Generating frequency-domain data...')
    tt = time.time()
    generate_seismic_data(shots_freq, solver, base_model, frequencies=frequencies, petsc='mumps')
    print('Frequency-domain data generation: {0}s'.format(time.time()-tt))



    # Check the result and plot the result
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
    





