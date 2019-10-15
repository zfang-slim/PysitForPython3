import numpy as np
import tensorflow as tf
from pysit.cnn.velocity_cnn import *

__all__ = ['Vel_CNN_Regularization']

class Vel_CNN_Regularization(object):
    """
        CNN net Regularization for velocity model

    """

    def __init__(self, CNN_Op, lambda=1.0, grad_to_vel=False):
        self.CNN_Op = CNN_Op
        self.lambda = lambda 
        self.grad_to_vel = grad_to_vel

    def __call__(self, m):
        m0 = m.data
        m0 = tf.convert_to_tensor(m0)
        y = self.CNN_Op.decoder_vel(m0)
        obj_val = 0.5*np.array(tf.math.reduce_sum(y*y))
        obj_val *= self.lambda 
        grad = self.CNN_Op.compute_decoder_derivative(m0)
        grad = np.array(grad)
        if grad_to_vel is False:
            grad = grad * (-0.5) * m.data**3.0
        grad *= self.lambda

        return obj_val, grad

        
