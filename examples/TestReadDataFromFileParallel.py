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

    Datafile = ExpDir + '/' + Datafile

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    if rank == 0:
        [data, odata, ddata, ndata] = read_data(Datafile)
        broad_info = {'o':odata, 'd':ddata, 'n':ndata}
        data1 = np.squeeze(data)
        for i in range(ndata[3]):
            print(i, 'th shot is ', np.linalg.norm(data1[:,:,i]))
    else:
        data = None
        broad_info = None 

    broad_info = comm.bcast(broad_info, root=0)
    if rank is not 0:
        odata = broad_info['o']
        ddata = broad_info['d']
        ndata = broad_info['n']

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 2.2, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 211, 71)

    shots = equispaced_acquisition_given_data(data, m, RickerWavelet(10.0),
                                             odata, ddata, ndata,
                                             parallel_shot_wrap=pwrap
                                             )

    #   Generate true wave speed
    
    for i in range(len(shots)):
        str = 'x shots position is '
        print('rank ', rank, '', i, 'th shot ', str, shots[i].sources.position)
        print('rank ', rank, ' ', i, 'th shot norm is', np.linalg.norm(shots[i].receivers.data))



