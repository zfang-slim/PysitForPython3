import numpy as np 
import copy 

__all__ = ['PMLExtensionPrj']

__docformat__ = "restructuredtext en"

class PMLExtensionPrj(object):
    ## The projection operator that projects a model with PML to a model with normal extension PML

    def __init__(self):
        self.name = 'PMLExtenstionPrj'

    def __call__(self, x):
        y = copy.deepcopy(x)
        y_tmp = y.without_padding()
        y = y_tmp.with_padding(padding_mode='edge')

        return y


if __name__ == '__main__':

    import time

    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import os
    from shutil import copy2

    import sys

    from pysit import *
    from pysit.gallery import horizontal_reflector
    from pysit.gallery.layered_medium import three_layered_medium
    from pysit.util.io import *

    n_mem = 1

    pmlz = PML(3, 100, ftype='quadratic')

#   pmlz = Dirichlet()

    z_config = (0.1, 0.15, pmlz, Dirichlet())
    z_config = (0.1, 0.15, pmlz, pmlz)
    nd = 5
#   z_config = (0.1, 0.8, Dirichlet(), Dirichlet())

    d = RectangularDomain(z_config)

    m = CartesianMesh(d, nd)

    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    Nshots = 1
    shots = []

    # Define source location and type
    zpos = 0.2
    source = PointSource(m, (zpos), RickerWavelet(25.0))

    # Define set of receivers
    receiver = PointReceiver(m, (zpos))
    # receivers = ReceiverSet([receiver])

    # Create and store the shot
    shot = Shot(source, receiver)
    # shot = Shot(source, receivers)
    shots.append(shot)

    # Define and configure the wave solver
    trange = (0.0, 3.0)

    solver1 = ConstantDensityAcousticWave(m,
                                          formulation='scalar',
                                          spatial_accuracy_order=2,
                                          trange=trange)

    solver2 = ConstantDensityAcousticWave(m,
                                          kernel_implementation='cpp',
                                          formulation='scalar',
                                          spatial_accuracy_order=2,
                                          trange=trange)

    # Generate synthetic Seismic data
    print('Generating data...')
    m_base = solver1.ModelParameters(m, {'C': C})
    c = copy.deepcopy(m_base)
    d = c - m_base
    c = m_base.with_padding()
    proj_op = PMLExtensionPrj()
    d = proj_op(c)

    a = 1




