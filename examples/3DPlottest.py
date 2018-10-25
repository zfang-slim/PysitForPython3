# Std import block
import time
import matplotlib as mpl
mpl.use('TkAgg')
# mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup
    os.environ["OMP_NUM_THREADS"] = "16"
    
    
    file_pi = open('ELSM_Result.obj', 'rb')
    x_out = pickle.load(file_pi)
    file_pi.close()

    axis_ticks = [[0, 10, 20, 30, 40], [5,20,35,50,65], [0, 20, 40]]
    axis_tickslabels = [[0, 100, 200, 300, 400],
                        [50, 200, 350, 500, 650],
                        [-200, 0, 200]]

    plt.figure()
    B = x_out.data
    B = np.reshape(B, (x_out.sh_sub[0], x_out.sh_sub[1], B.shape[1]))
    # B = np.ones((45,71,41))*0.5
    # B[:,0:30,:] = 0.0
    line_location = [24, 40, 20]
    vis.plot_extend_image_2D(x_out, line_location=line_location, vmin=-20.0, vmax=20.0)
    # vis.plot_3D_panel(B, slice3d=(24, 40, 20), width_ratios=[71, 21], height_ratios=[21, 71], cmap='gray', vmin=-20.0, vmax=20.0,
    #                   axis_label=['x [m]', 'z [m]', 'h [m]'],
    #                   axis_ticks=axis_ticks,
    #                   axis_tickslabels=axis_tickslabels,
    #                   line_location=line_location)
    # plt.show()
    plt.savefig("D.png")

    a =1

