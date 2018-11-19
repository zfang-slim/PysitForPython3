import numpy as np

import scipy.io as sio

import obspy.io.segy.core as segy

__all__ = ['odn2grid', 'odn2grid_data_2D_time', 'odn2grid_data_3D_time',
           'odn2grid_data_2D_freq', 'odn2grid_data_3D_freq']

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

        



