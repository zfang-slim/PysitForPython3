import numpy as np 
import os 
import pkg_resources
import pykonal 
import pandas as pd
import copy as copy
from pysit.cnn.velocity_cnn import Vel_CNN_Overthrust
from pysit.util.parallel import ParallelWrapShotNull

__all__ = ['TomoObj','SourceTomo','TomoObjFun']

__docformat__ = "restructuredtext en"

class SourceTomo(object):
    def __init__(self, source_pos, rec_x, rec_z, rec_y=None):
        self.source_pos = source_pos
        if rec_y is None:
            rec_y = np.zeros(len(rec_x),)
        self.receivers  = {'x':rec_x,
                           'y':rec_y,
                           'z':rec_z}

class TomoObj(object):
    def __init__(self, min_coords, intervals, ngrids, sources):
        """
            The order of coordinate is x, z, y
        """
        if len(min_coords) == 2:
            min_coords = [min_coords[0], min_coords[1], 0]
        if len(intervals) == 2:
            intervals = [intervals[0], intervals[1], 1]
        if len(ngrids) == 2:
            ngrids = [ngrids[0], ngrids[1], 1]
         
        self.min_coords = min_coords
        self.intervals = intervals
        self.ngrids = ngrids
        self.solver_tomo = pykonal.EikonalSolver()
        self.solver_tomo.vgrid.min_coords = min_coords
        self.solver_tomo.vgrid.node_intervals = intervals
        self.solver_tomo.vgrid.npts = ngrids

        self.solver_tomo.pgrid.min_coords = min_coords
        self.solver_tomo.pgrid.node_intervals = intervals
        self.solver_tomo.pgrid.npts = ngrids
        self.sources = sources

    def forward_map(self, m):
        # input m should be velocity 
        output = []
        m = np.reshape(m, self.ngrids)
        self.solver_tomo.vv = m
        for isrc in range(len(self.sources)):
            source = self.sources[isrc]
            solver = copy.deepcopy(self.solver_tomo)
            solver.add_source(source.source_pos)
            solver.solve()
            ui = pykonal.LinearInterpolator3D(solver.pgrid, solver.uu)
            xrc = source.receivers['x']
            yrc = source.receivers['z']
            zrc = source.receivers['y']
            rc  = pd.DataFrame({'x':xrc,
                                'y':yrc,
                                'z':zrc})
            rec_data = np.zeros([len(xrc), 1])
            for irec in range(len(xrc)):
                rec_data[irec] = ui(rc.iloc[irec])

            output.append(rec_data)

        return output

class TomoObjFun(object):
    def __init__(self, tomo_obj, data_obs, sigma, parallel_wrap_shot=ParallelWrapShotNull(), cnn=None):
        self.data_obs = data_obs
        self.tomo_obj = tomo_obj
        self.sigma = sigma
        self.cnn = cnn

    def evaluate(self, m):
        m_cmp = m
        if self.cnn is not None:
            m_cmp = self.cnn.generate_vel(m)
            m_cmp = np.array(m_cmp)
            m_cmp = np.reshape(m_cmp, self.tomo_obj.ngrids)
            m_cmp = m_cmp.transpose()

        data_obs = self.data_obs
        sigma = self.sigma
        data_pred = self.tomo_obj.forward_map(m_cmp)
        obj_val = 0.0
        for i in range(len(data_pred)):
            obj_val_i = np.linalg.norm(data_pred[i] - data_obs[i])**2.0 / 2.0
            obj_val += obj_val_i
        obj_val /= sigma**2.0 

        return obj_val


            
if __name__ == '__main__':
    # from pysit import *
    
    source_pos = [[2, 1, 0], [4, 1, 0]]
    rec_x = np.linspace(1,8,10)
    rec_z = np.ones(10,)*2.5
    rec_y = np.zeros(10,)
    sources = []
    for i in range(len(source_pos)):
        sources.append(SourceTomo(source_pos[i], rec_x, rec_z, rec_y))
    
    file = '/Users/zhilongfang/Documents/Git-Hub/pykonal/pykonal/data/marmousi_2d.npz'
    with np.load(file) as infile:
        vv = infile['vv']
    min_coords = 0,0,0 
    intervals = 0.004, 0.004, 1 
    ngrids = vv.shape

    TomoTest = TomoObj(min_coords, intervals, ngrids, sources)
    r_out = TomoTest.forward_map(vv)
    Tomo_objfun = TomoObjFun(TomoTest, r_out, 1)
    v0 = np.ones(ngrids, dtype='float32')
    obj = Tomo_objfun.evaluate(v0)

    print('done')
    print(r_out)
    print(obj)