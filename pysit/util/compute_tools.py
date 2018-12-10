import numpy as np

import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt

import obspy.io.segy.core as segy




__all__ = ['odn2grid', 'odn2grid_data_2D_time', 'odn2grid_data_3D_time',
           'odn2grid_data_2D_freq', 'odn2grid_data_3D_freq', 'low_pass_filter']

def odn2grid(o, d, n):
    output = dict()
    for i in range(len(o)):
        tmp = np.array(range(0, n[i])).astype(float)
        output[i] = o[i] + tmp*d[i]

    return output

def odn2grid_data_2D_time(o, d, n):
    output = odn2grid(o, d, n)
    data_time = output[0]
    data_xrec = output[1]
    data_zrec = output[2]
    data_xsrc = output[3]
    data_zsrc = output[4]

    return data_time, data_xrec, data_zrec, data_xsrc, data_zsrc


def odn2grid_data_3D_time(o, d, n):
    output = odn2grid(o, d, n)
    data_time = output[0]
    data_xrec = output[1]
    data_yrec = output[2]
    data_zrec = output[3]
    data_xsrc = output[4]
    data_ysrc = output[5]
    data_zsrc = output[6]

    return data_time, data_xrec, data_yrec, data_zrec, data_xsrc, data_ysrc, data_zsrc


def odn2grid_data_2D_freq(o, d, n):
    output = odn2grid(o, d, n)
    data_xrec = output[0]
    data_zrec = output[1]
    data_xsrc = output[2]
    data_zsrc = output[3]
    data_freq = output[4]

    return data_xrec, data_zrec, data_xsrc, data_zsrc, data_freq


def odn2grid_data_3D_freq(o, d, n):
    output = odn2grid(o, d, n)
    data_xrec = output[0]
    data_yrec = output[1]
    data_zrec = output[2]
    data_xsrc = output[3]
    data_ysrc = output[4]
    data_zsrc = output[5]
    data_freq = output[6]

    return data_xrec, data_yrec, data_zrec, data_xsrc, data_ysrc, data_zsrc, data_freq

class low_pass_filter(object):
    ''' This is a low pass filter object that conducts the 1D low pass filtering
        
    '''

    def __init__(self, nsmp, T, cut_freq, transit_freq_length=1.0, axis=0):
        '''
        Input:
            data - data can be a 1D array or 2D matrix or 3D cubic
            nsmp - number of sample points
            T - the maximum physical time of the signal
            cut_freq - the frequency that uses to cut the signal
            transit_freq_length - the length of frequency that transit the filter value from 0 to 1
            axis - the axis over which to compute the low pass filter, default is -1

        output:
            dataout - the data after the low pass filter
        '''

        self.T = T 
        self.nsmp = nsmp
        self.cut_freq = cut_freq
        self.transit_freq_length = transit_freq_length
        self.axis = axis
        
        df = 1/T
        self.df = df

        n_cut = int((cut_freq+transit_freq_length) / df) + 1
        low_pass_filter = np.ones(nsmp)
        low_freq_part = np.zeros(n_cut)
        ind_transit_start = n_cut - int(transit_freq_length / df) 
        ind_transit_end = n_cut - 1
        n_transit = ind_transit_end - ind_transit_start + 1

        x_transit = np.linspace(0.0, np.pi/2.0, n_transit)
        y_transit = np.sin(x_transit)

        low_freq_part[n_cut-n_transit : n_cut] = y_transit
        high_freq_part = np.flip(low_freq_part, axis=-1)

        low_pass_filter[0:n_cut] = low_freq_part
        low_pass_filter[nsmp-n_cut:nsmp] = high_freq_part

        self.low_pass_filter = low_pass_filter 

    def __mul__(self, x):
        
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.nsmp:
                raise ValueError('The size of input x should be equal to the nsmp of the low pass filter object')

            else:
               y = self._apply_filter(x) 

        else:
            if x_shape[self.axis] != self.nsmp:
                raise ValueError("The length of input x's operating axis should be equal to the nsmp of the low pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y




    def _apply_filter(self, x):
        y = np.fft.fft(x)
        y = self.low_pass_filter * y
        y = np.fft.ifft(y)
        return y.real

        





if __name__ == '__main__':
    o = [0.0, 1.0, 2.0]
    d = [1.0, 2.0, 3.0]
    n = [4, 5, 6]
    output = odn2grid(o, d, n)
    print(output[0])
    print(output[1])
    print(output[2])

    o = [0.0, 1.0, 2.0, 3.0, 4.0]
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = [4, 1, 5, 1, 6]
    data_time, data_xrec, data_zrec, data_xsrc, data_zsrc = odn2grid_data_2D_time(o, d, n)

    print(data_time)
    print(data_zrec)
    print(data_xrec)
    print(data_zsrc)
    print(data_xsrc)

    nsmp = 101
    dt = 0.1
    T = (nsmp-1)*dt 
    xt = np.linspace(0, T, nsmp)
    a = 4.0
    f0 = signal.ricker(nsmp, a)
    cut_freq = 0.5


    LF = low_pass_filter(nsmp, T, cut_freq)
    f1 = LF * f0

    plt.plot(xt,f0)
    plt.plot(xt,f1)
    plt.show()
    plt.figure()
    plt.plot(np.abs(np.fft.fft(f0)))
    plt.plot(np.abs(np.fft.fft(f1)))
    plt.show()

    f0=f0.reshape((-1,1))
    F0 = np.concatenate((f0,f0),axis=1)
    F1 = LF * F0

    a = 1

        



