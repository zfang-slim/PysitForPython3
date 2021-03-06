import numpy as np 
import copy 
import tensorflow as tf

__all__ = ['PMLExtensionPrj', 'BoxConstraintPrj', 'WaterVelocityPrj', 'JointPrj', 'Unet_Vel_OverthrustPrj']

__docformat__ = "restructuredtext en"

class Unet_Vel_OverthrustPrj(object):
    """
        Unet projector trained by overthrust model

    """

    def __init__(self, CNN_Op):
        self.CNN_Op = CNN_Op

    def __call__(self, m):
        output = copy.deepcopy(m)
        m0 = tf.convert_to_tensor(m.data, dtype=tf.float32)
        y = self.CNN_Op.decoder_vel(m0)
        m_out = self.CNN_Op.generate_vel(y)
        m_out_1 = np.array(m_out).flatten()
        m_out_1 = np.reshape(m_out_1, np.shape(output.data))
        output.data = m_out_1

        return output       

class PMLExtensionPrj(object):
    ## The projection operator that projects a model with PML to a model with normal extension PML

    def __init__(self):
        self.name = 'PMLExtenstionPrj'

    def __call__(self, x):
        y = copy.deepcopy(x)
        y_tmp = y.without_padding()
        y = y_tmp.with_padding(padding_mode='edge')

        return y


class BoxConstraintPrj(object):
    ## The projection operator that conducts the box constraint

    def __init__(self, bound, pointwise=False):
        self.name = 'BoxConstraint'
        self.bound = bound
        self.pointwise = pointwise

    def __call__(self, x):
        y = copy.deepcopy(x)
        if self.pointwise is not True:
            idx = np.where(y.data < self.bound[0])
            y.data[idx] = self.bound[0]
            idx = np.where(y.data > self.bound[1])
            y.data[idx] = self.bound[1]
            idx = np.where(np.isnan(y.data))
            y.data[idx] = self.bound[1]
        else:
            for i in range(len(y.data)):
                if y.data[i] < self.bound[i,0]:
                    y.data[i] = self.bound[i,0]
                elif y.data[i] > self.bound[i,1]:
                    y.data[i] = self.bound[i,1]

        return y

class WaterVelocityPrj(object):

    def __init__(self, ModelSize, NumOfWaterLayer=1, WaterVel=1.5):
        self.name = 'WaterVelocity'
        self.NumOfWaterLayer = NumOfWaterLayer
        self.WaterVel = WaterVel
        self.ModelSize = ModelSize
        self.dim = len(ModelSize)

    def __call__(self, x):
        y = copy.deepcopy(x)
        z = np.reshape(y.data, self.ModelSize)
        if self.dim == 1:
            z[0:self.NumOfWaterLayer] = self.WaterVel
        elif self.dim == 2:
            z[:, 0:self.NumOfWaterLayer] = self.WaterVel
        else:
            z[:,:,0:self.NumOfWaterLayer] = self.WaterVel

        y.data = np.reshape(z, y.data.shape)


        return y
        

class JointPrj(object):
    ## The project that conducts the projection onto the intersection of two constraints

    def __init__(self, proj1, proj2, niter=300, epsilon=10.0**(-6.0)):
        self.name = proj1.name + '+' + proj2.name
        self.proj1 = proj1
        self.proj2 = proj2
        self.niter = niter
        self.epsilon = epsilon

    def __call__(self, x):
        iter = 1
        pk = x - x
        qk = x - x

        stop = False
        xk = copy.deepcopy(x)

        while stop is False :
            yk = self.proj1(xk + pk)
            pkp1 = (xk + pk) - yk
            xkp1 = self.proj2(yk + qk) 
            qkp1 = (yk + qk) - xkp1

            if (xkp1-xk).inner_product(xkp1-xk) < self.epsilon:
                stop = True
            else:
                iter += 1
                if iter > self.niter:
                    stop = True
                else:
                    xk = xkp1 
                    pk = pkp1
                    qk = qkp1
 

        return xkp1








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

    z_config = (0.1, 0.61, pmlz, Dirichlet())
    z_config = (0.1, 0.61, pmlz, pmlz)
    nd = 51
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

    proj_op1 = proj_op
    proj_op2 = BoxConstraintPrj([1.6, 3.0])
    proj_op_joint = JointPrj(proj_op1, proj_op2)

    d.data = np.linspace(1.5,3.5,len(d.data))
    e = proj_op_joint(d)

    plt.plot(d.data)
    plt.plot(e.data)
    plt.show()

    # proj_op3 = WaterVelocityPrj()


    a = 1




