import copy

import numpy as np

from mpi4py import MPI

from .shot import *

from .receivers import *
from .sources import *

from pysit.util.parallel import ParallelWrapShotNull
from pysit.util.compute_tools import *

__all__ = ['equispaced_acquisition',
           'equispaced_acquisition_given_data']

def equispaced_acquisition(mesh, wavelet,
                           sources=1,
                           receivers='max',
                           source_depth=None,
                           source_kwargs={},
                           receiver_depth=None,
                           receiver_kwargs={},
                           parallel_shot_wrap=ParallelWrapShotNull()
                           ):

    m = mesh
    d = mesh.domain

    xmin = d.x.lbound
    xmax = d.x.rbound

    zmin = d.z.lbound
    zmax = d.z.rbound

    if m.dim == 3:
        ymin = d.y.lbound
        ymax = d.y.rbound


    if source_depth is None:
        source_depth = zmin

    if receiver_depth is None:
        receiver_depth = zmin

    shots = list()

    max_sources = m.x.n

    if m.dim == 2:
        if receivers == 'max':
            receivers = m.x.n
        if sources == 'max':
            sources = m.x.n

        if receivers > m.x.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')
        if sources > m.x.n:
            raise ValueError('Number of sources exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers)
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, receiver_depth), **receiver_kwargs) for x in xpos])

        local_sources = sources / parallel_shot_wrap.size

    if m.dim == 3:

        if receivers == 'max':
            receivers = (m.x.n, m.y.n) # x, y
        if sources == 'max':
            sources = (m.x.n, m.y.n) # x, y

        if receivers[0] > m.x.n or receivers[1] > m.y.n:
            raise ValueError('Number of receivers exceeds mesh nodes.')
        if sources[0] > m.x.n or sources[1] > m.y.n:
            raise ValueError('Number of sources exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers[0])
        ypos = np.linspace(ymin, ymax, receivers[1])
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, y, receiver_depth), **receiver_kwargs) for x in xpos for y in ypos])

        local_sources = np.prod(sources) / parallel_shot_wrap.size
        

    print(type(local_sources))
    print(local_sources)

    for k in range(int(local_sources)):
        index_true = int(local_sources) * parallel_shot_wrap.rank + k
        subindex = np.unravel_index(index_true, sources)
        idx = subindex[0]

        if m.dim == 3:
            jdx = subindex[1]

        if m.dim == 2:
            srcpos = (xmin + (xmax-xmin)*(idx+1.0)/(sources+1.0), source_depth)
        elif m.dim == 3:
            srcpos = (xmin + (xmax-xmin)*(idx+1.0)/(sources[0]+1.0), ymin + (ymax-ymin)*(jdx+1.0)/(sources[1]+1.0), source_depth)

        # Define source location and type
        source = PointSource(m, srcpos, wavelet, **source_kwargs)

        # Define set of receivers
        receivers = copy.deepcopy(receiversbase)

        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)

    return shots

def equispaced_acquisition_given_data(data, mesh, wavelet,
                                      odata, ddata, ndata,
                                      source_kwargs={},
                                      receiver_kwargs={},
                                      parallel_shot_wrap=ParallelWrapShotNull()
                                      ):


    source_depth=None,
    receiver_depth=None,

    m = mesh
    d = mesh.domain

    xmin = d.x.lbound
    xmax = d.x.rbound

    zmin = d.z.lbound
    zmax = d.z.rbound

    if m.dim == 2:
        data_time, data_xrec, data_zrec, data_xsrc, data_zsrc = odn2grid_data_2D_time(odata, ddata, ndata)

    if m.dim == 3:
        data_time, data_xrec, data_yrec, data_zrec, data_xsrc, data_ysrc, data_zsrc = odn2grid_data_3D_time(odata, ddata, ndata)

    if m.dim == 3:
        ymin = d.y.lbound
        ymax = d.y.rbound

    source_depth = data_zsrc[0]
    receiver_depth = data_zrec[0] 

    shots = list()

    max_sources = m.x.n

    if m.dim == 2:
        receivers = ndata[1]
        sources = ndata[3]

        xpos_rec = data_xrec
        
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, receiver_depth), **receiver_kwargs) for x in xpos_rec])

        
        if np.mod(sources, parallel_shot_wrap.size) != 0:
            raise ValueError('Currently, we only support the case that mod(number of sources, number of processes) = 0')
        local_sources = sources / parallel_shot_wrap.size

    if m.dim == 3:

        receivers = (ndata[1], ndata[2])
        sources = (ndata[4], ndata[5])
        
        xpos_rec = data_xrec
        ypos_rec = data_yrec
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, y, receiver_depth), **receiver_kwargs) for x in xpos_rec for y in ypos_rec])

        if np.mod(np.prod(sources), parallel_shot_wrap.size) != 0:
            raise('Currently, we only support the case that mod(number of sources, number of processes) = 0')
        local_sources = np.prod(sources) / parallel_shot_wrap.size

    print(type(local_sources))
    local_sources = int(local_sources)

    if m.dim == 2:
        if parallel_shot_wrap.rank == 0:
            data_local = data[:,:,:,0:local_sources,:].squeeze()

            for i in range(1, parallel_shot_wrap.size):
                data_send = data[:,:,:,i*local_sources:(i+1)*local_sources,:]
                parallel_shot_wrap.comm.send(data_send, dest=i, tag=i)

        else:
            data_receive=parallel_shot_wrap.comm.recv(source=0, tag=parallel_shot_wrap.rank)
            print('Receive data from process ', 0)

            data_local = data_receive.squeeze()

    if m.dim == 3:
        if parallel_shot_wrap.rank == 0:
            data_local = get_local_data(data, n, local_sources, 0)
            for k in range(1, parallel_shot_wrap.size):
                data_send = get_local_data(data, n, local_source, k)
                parallel_shot_wrap.comm.send(data_send, dest=k, tag=k)

        else:
            data_local=parallel_shot_wrap.comm.recv(source=0, tag=parallel_shot_wrap.rank)
            print('Receive data from process ', 0)
            # data_local = np.zeros((data_time, data_xrec*data_yrec, local_sources))
            # for k in range(local_sources):


    for k in range(int(local_sources)):
        index_true = int(local_sources) * parallel_shot_wrap.rank + k
        subindex = np.unravel_index(index_true, sources)

        if m.dim == 2:
            idx = subindex
        
        if m.dim == 3:
            idx = subindex[0]
            jdx = subindex[1]

        if m.dim == 2:
            srcpos = (data_xsrc[idx], source_depth)
        elif m.dim == 3:
            srcpos = (data_xsrc[idx], data_ysrc[jdx], source_depth)

        # Define source location and type
        source = PointSource(m, srcpos, wavelet, **source_kwargs)

        # Define set of receivers
        receivers = copy.deepcopy(receiversbase)

        receivers.data = data_local[:,:,k]

        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)


    return shots

def get_local_data(data, n, local_sources, rank):
    n_out = (n[0], n[1]*n[2], local_sources)
    data_out = np.zeros(n_out)

    for k in range(local_sources):
        indx_k = rank * local_sources + k
        indx_sub = np.unravel_index(indx_k, (n[4], n[5]))
        data_tmp = np.reshape(data[:,:,:,:,indx_sub[0],indx_sub[1],:], (n[0], n[1]*n[2]))
        data_out[:,:,k] = data_tmp

    return data_out
            
        

