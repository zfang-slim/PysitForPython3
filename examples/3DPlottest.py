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
    
    
    file_pi = open('ELSM_True.obj', 'rb')
    x_out = pickle.load(file_pi)
    file_pi.close()

    plt.figure()
    B = x_out.data
    B = np.reshape(B, (x_out.sh_sub[0], x_out.sh_sub[1], B.shape[1]))
    vis.plot_3d_panel(B, slice3d=(20, 40, 20), width_ratios=[71, 21], height_ratios=[21, 71], cmap='gray', vmin=-1.0, vmax=1.0,
                      axis_label=['x [m]', 'z [m]', 'h [m]'])
    plt.show()
    plt.savefig("D.png")

    a =1

