import numpy as np
import tensorflow as tf
from pysit.cnn.velocity_cnn import *

__all__ = ['Vel_CNN_Regularization', 'Vel_CNN_Regularization2']

class Vel_CNN_Regularization(object):
    """
        CNN net Regularization for velocity model

    """

    def __init__(self, CNN_Op, alpha=1.0, grad_to_vel=False):
        self.CNN_Op = CNN_Op
        self.alpha = alpha 
        self.grad_to_vel = grad_to_vel

    def __call__(self, m):
        m0 = tf.convert_to_tensor(m, dtype=tf.float32)
        y = self.CNN_Op.decoder_vel(m0)
        obj_val = 0.5*np.array(tf.math.reduce_sum(y*y))
        obj_val *= self.alpha 
        grad = self.CNN_Op.compute_decoder_derivative(m0, y)
        grad = np.array(grad)
        if self.grad_to_vel is False:
            grad = grad * (-0.5) * m**3.0
        grad *= self.alpha

        grad = np.reshape(np.shape(m))

        return obj_val, grad

class Vel_CNN_Regularization2(object):
    """
        CNN net Regularization for velocity model

    """

    def __init__(self, CNN_Op, alpha=1.0, grad_to_vel=False):
        self.CNN_Op = CNN_Op
        self.alpha = alpha 
        self.grad_to_vel = grad_to_vel

    def __call__(self, m):
        m_input = m.flatten()
        m0 = tf.convert_to_tensor(m, dtype=tf.float32)
        y = self.CNN_Op.decoder_vel(m0)
        m_out = self.CNN_Op.generate_vel(y)
        m_out_1 = np.array(m_out).flatten()
        res = tf.reshape(m_out, tf.shape(m0)) - m0
        res_1 = m_out_1 - m_input

        obj_val = 0.5*np.linalg.norm(res_1)**2.0
        obj_val *= self.alpha 
        grad1 = -res_1
        grad2 = self.CNN_Op.compute_generator_derivative(y, res)
        grad2 = self.CNN_Op.compute_decoder_derivative(m0, grad2)
        grad = grad1 + np.array(grad2).flatten()
        # grad = self.CNN_Op.compute_decoder_derivative(m0)
        # grad = np.array(grad)
        if self.grad_to_vel is False:
            grad = grad * (-0.5) * m_input**3.0
        grad *= self.alpha

        grad = np.reshape(np.shape(m))

        return obj_val, grad        
