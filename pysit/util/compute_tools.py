import numpy as np

import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt

import obspy.io.segy.core as segy




__all__ = ['odn2grid', 'odn2grid_data_2D_time', 'odn2grid_data_3D_time',
           'odn2grid_data_2D_freq', 'odn2grid_data_3D_freq', 'low_pass_filter',
           'high_pass_filter', 'band_pass_filter', 'correlate_fun', 'optimal_transport_fwi']

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

class high_pass_filter(object):
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
        high_pass_filter = np.ones(nsmp)
        low_freq_part = np.zeros(n_cut)
        ind_transit_start = n_cut - int(transit_freq_length / df) 
        ind_transit_end = n_cut - 1
        n_transit = ind_transit_end - ind_transit_start + 1

        x_transit = np.linspace(0.0, np.pi/2.0, n_transit)
        y_transit = np.sin(x_transit)

        low_freq_part[n_cut-n_transit : n_cut] = y_transit
        high_freq_part = np.flip(low_freq_part, axis=-1)

        high_pass_filter[0:n_cut] = low_freq_part
        high_pass_filter[nsmp-n_cut:nsmp] = high_freq_part

        self.high_pass_filter = high_pass_filter 

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
        y = self.high_pass_filter * y
        y = np.fft.ifft(y)
        return y.real


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
        low_pass_filter = np.zeros(nsmp)
        low_freq_part = np.ones(n_cut)
        ind_transit_start = n_cut - int(transit_freq_length / df)
        ind_transit_end = n_cut - 1
        n_transit = ind_transit_end - ind_transit_start + 1

        x_transit = np.linspace(0.0, np.pi/2.0, n_transit)
        y_transit = np.cos(x_transit)

        low_freq_part[n_cut-n_transit: n_cut] = y_transit
        high_freq_part = np.flip(low_freq_part, axis=-1)

        low_pass_filter[0:n_cut] = low_freq_part
        low_pass_filter[nsmp-n_cut:nsmp] = high_freq_part

        self.low_pass_filter = low_pass_filter

    def __mul__(self, x):

        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.nsmp:
                raise ValueError(
                    'The size of input x should be equal to the nsmp of the low pass filter object')

            else:
               y = self._apply_filter(x)

        else:
            if x_shape[self.axis] != self.nsmp:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the nsmp of the low pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y

    def _apply_filter(self, x):
        y = np.fft.fft(x)
        y = self.low_pass_filter * y
        y = np.fft.ifft(y)
        return y.real


class band_pass_filter(object):
    ''' This is a low pass filter object that conducts the 1D low pass filtering
        
    '''

    def __init__(self, nsmp, T, freq_band, transit_freq_length=1.0, axis=0):
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
        self.freq_band = freq_band
        self.transit_freq_length = transit_freq_length
        self.axis = axis

        df = 1/T
        self.df = df

        LPF = low_pass_filter(nsmp, T, freq_band[1], transit_freq_length=transit_freq_length, axis=axis)
        HPF = high_pass_filter(nsmp, T, freq_band[0], transit_freq_length=transit_freq_length, axis=axis)

        self.band_pass_filter = LPF.low_pass_filter * HPF.high_pass_filter

    def __mul__(self, x):

        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.nsmp:
                raise ValueError(
                    'The size of input x should be equal to the nsmp of the low pass filter object')

            else:
               y = self._apply_filter(x)

        else:
            if x_shape[self.axis] != self.nsmp:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the nsmp of the low pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y

    def _apply_filter(self, x):
        y = np.fft.fft(x)
        y = self.band_pass_filter * y
        y = np.fft.ifft(y)
        return y.real

def correlate_fun(dobs, dpred, mode='fwd'):
    
    nd = len(dobs)
    if mode == 'fwd':
        output = np.correlate(dobs, dpred, mode='full')

    else:
        ndobs = len(dobs)
        ndpred = len(dpred)
        output = np.zeros(ndobs)
        
        for i in range(ndobs):
            output[i] = np.dot(dobs, dpred[ndpred-ndobs-i:ndpred-i])


    return output

def optimal_transport_fwi(dobs, dpred, dt):
    
    # Normalization and transfer data to a distribution
    c = 2.0 * np.abs(np.min(dobs))
    g = dobs + c
    g = g / (np.sum(g)*dt)
    f_plus_c = dpred + c
    s = (np.sum(f_plus_c)*dt)
    f = f_plus_c / s
    ndata = len(f)
    t = np.array(range(0,ndata)) * dt
    
    print('min f {0}'.format(np.min(f)))
    print('min g {0}'.format{np.min(g)})

    F = np.zeros(ndata)
    G = np.zeros(ndata)
    IGoF = np.zeros(ndata)
    IGoF_ind = np.zeros(ndata)

    # Compute F(t) and G(t)
    int_f = 0.0
    int_g = 0.0
    for i in range(0, ndata):
        int_f += f[i]
        int_g += g[i]
        F[i] = int_f
        G[i] = int_g

    # Compute G^{-1} o F(t)
    IGoF[ndata-1] = (ndata-1)*dt
    IGoF_ind[ndata-1] = ndata-1
    for i in range(1, ndata-1):
        IGoF_ind[i] = np.searchsorted(G, F[i])
        IGoF[i] = IGoF_ind[i] * dt

    IGoF_ind = IGoF_ind.astype(int)

    idx = np.where(IGoF_ind>ndata-1)
    IGoF_ind[idx] = ndata-1

    # Compute residual
    t_minus_IGoF = t - IGoF
    resid = np.sqrt(f) * t_minus_IGoF

    # Compute adjoint source

    adj_src1 = t_minus_IGoF * t_minus_IGoF
    adj_src2 = (-2*f*dt / g[IGoF_ind]) * t_minus_IGoF

    for i in range(ndata-2, -1, -1):
        adj_src2[i] += adj_src2[i+1]

    adj_src = adj_src1+adj_src2
    adj_src = adj_src / s - (dt/(s**2.0)*np.dot(f_plus_c, adj_src))*np.ones(ndata)

    return resid, adj_src, np.linalg.norm(resid)**2.0


if __name__ == '__main__':
    
    import numpy as np
    
    a = np.array([0,0,-1,0,1,0,-1,0,0])
    b = np.array([0,-1,0,1,0,-1,0,0,0])
    dt = 0.2
    resid, adj_src, r = optimal_transport_fwi(a, b, dt)
    
    nd = 11
    a = np.random.normal(0,1,nd)
    b = np.random.normal(0,1,nd*2-1)
    c = np.random.normal(0,1,nd)
    
    d = np.dot(b, correlate_fun(a,c))
    e = np.dot(c, correlate_fun(a,b,mode='adj'))
    print('d = {0}'.format(d))
    print('e = {0}'.format(e))
    
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
    freqt = np.linspace(0, 10, nsmp)
    a = 4.0
    f0 = signal.ricker(nsmp, a)
    cut_freq = 0.2
    cut_freqh = 0.5
    freq_band = [0.2, 1.0]


    LF = low_pass_filter(nsmp, T, cut_freq)
    f1 = LF * f0

    plt.plot(xt,f0)
    plt.plot(xt,f1)
    plt.show()
    plt.figure()
    plt.plot(freqt, np.abs(np.fft.fft(f0)))
    plt.plot(freqt, np.abs(np.fft.fft(f1)))
    plt.show()

    HF = high_pass_filter(nsmp, T, cut_freqh)
    f1 = HF * f0

    plt.plot(xt, f0)
    plt.plot(xt, f1)
    plt.show()
    plt.figure()
    plt.plot(freqt, np.abs(np.fft.fft(f0)))
    plt.plot(freqt, np.abs(np.fft.fft(f1)))
    plt.show()

    BF = band_pass_filter(nsmp, T, freq_band)
    f1 = BF * f0

    plt.plot(xt, f0)
    plt.plot(xt, f1)
    plt.show()
    plt.figure()
    plt.plot(freqt, np.abs(np.fft.fft(f0)))
    plt.plot(freqt, np.abs(np.fft.fft(f1)))
    plt.show()


    f0=f0.reshape((-1,1))
    F0 = np.concatenate((f0,f0),axis=1)
    F1 = LF * F0

    a = 1

        



