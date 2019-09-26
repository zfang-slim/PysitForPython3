import tensorflow as tf
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
# import PIL
from tensorflow.keras import layers
import time
import scipy.io as sio

__all__ = ['Vel_CNN_Overthrust']

class Vel_CNN_Overthrust(object):
    """
        CNN net for velocity model generation

    """

    def __init__(self, weights_file):

        # Define the generator and discriminator
        generator = self.make_generator_model()
        discriminator = self.make_discriminator_model()

        # Load weights_file and set the weights to generator and discriminator
        # For convience, we only save checkpoint and restor checkpoint
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)

        checkpoint.restore(weights_file)
        
        self.generator = generator
        self.discriminator = discriminator

    def generate_vel(self, m, training=False, a=1911.0, b=2989.0):
        y = self.generator(m, training=training)
        y = y * a + b
        y = y / 1000.0
        return y

    def compute_derivative(self, m, gradient_v):
        with tf.GradientTape() as gen_tape:
            gen_tape.watch(m)
            yy = self.generate_vel(m, training=False)
            yy = tf.reshape(yy, (np.prod(yy.shape),1))
            f = tf.math.reduce_sum(yy*gradient_v)
            g = gen_tape.gradient(f, m)
            

        return g

    @staticmethod
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(16*16*128, use_bias=False, input_shape=(50,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((16, 16, 128)))
        assert model.output_shape == (None, 16, 16, 128) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 1)

        return model

    @staticmethod
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[64, 64, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model



if __name__ == '__main__':
    from velocity_cnn import *
    checkpoint = '/Users/zhilongfang/GoogleDriver/Data/TenLayers/training_checkpoints/ckpt-19'

    CNN_Vel = Vel_CNN_Overthrust(checkpoint)




