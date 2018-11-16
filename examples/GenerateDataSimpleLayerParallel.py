# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from shutil import copy2

import sys
import scipy.io as sio

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.io import *

from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup

    RootDir = '/Users/fangzl/Data/Data'
    # RootDir = '/wavedata/Zhilong/ExxonProject/LayerModel/Data'
    SubDir = '/Layer_FWI1'
    Datafile = 'LayerData2.mat'

    ExpDir = RootDir + SubDir

    if not os.path.exists(ExpDir):
        os.mkdir(ExpDir)
        print("Dirctory ", ExpDir, " Created")

    currentfile = os.path.basename(__file__)
    copy2(currentfile, ExpDir)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()


    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 2.2, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 211, 71)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    Nshots = size * 2
    sys.stdout.write("{0}: {1}\n".format(rank, Nshots / size))

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


    # Define and configure the wave solver
    trange = (0.0,0.1)
    ts = np.linspace(trange[0], trange[1], 751)
    dts = ts[1] - ts[0]

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    sys.stdout.write('Generating data...')
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)

    print('rank is ', rank)
    print('size is ', size)
    
    if rank == 0:
        print('Run time:  {0}s'.format(time.time()-tt))

    print('The length of shot equals ', len(shots))
    shots_all = comm.gather(shots, root=0)

    # if rank == 0
    #     n_data = (shots[0].receivers.data.shape[0], 1, Nshots, 1, shots[0].receivers.data.shape[1])
    #     o_data = (0, 1, )
    

    if rank == 0:
        print('The length of shots_all equals ', len(shots_all))
        print(shots_all)
        write_gathered_parallel_data_time(ExpDir + '/' + Datafile, shots_all)







    # Do something to visualize the results
#   display_on_grid(C, d, shade_pml=True)
#   display_on_grid(result.C, d, shade_pml=True)
    #display_seismogram(shots[0], clim=[-1,1])
    #display_seismogram(shots[0], wiggle=True, wiggle_skip=1)
    # animate_wave_evolution(ps, domain=d, display_rate=10, shade_pml=True)
