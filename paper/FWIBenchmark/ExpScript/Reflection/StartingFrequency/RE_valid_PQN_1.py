import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
from pysit.util.util import *

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import copy as copy
import math
import os
import scipy.io as sio
from scipy.signal import hilbert
from shutil import copy2

import sys

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.gallery.layered_medium import three_layered_medium
from pysit.util.io import *
from pysit.util.compute_tools import *

from pysit.util.parallel import *

from mpi4py import MPI

def creat_model(n, c_w, n_conv, vb, vp):
    s1 = (n[0]-c_w[0])//2
    s2 = (n[1]-c_w[1])//2
    A = np.ones(n)*vb
    A[s1:n[0]-s1,s2:n[1]-s2] = vp
    S = opSmooth2D(n, n_conv, window_len=[5, 5])
    B = S * A.flatten()
    B = B.reshape(n)

    return B

def create_model3(n, vel):
    v1 = np.ones((n[0],1))
    k = n[0] // 2
    v1[0:k] = vel[0]
    v1[k:2*k-5] = vel[1]
    v1[2*k-5:n[0]]= vel[2]
    v1 = v1.reshape([n[0],1])

    B  = np.matlib.repmat(v1,1,n[1])
    print(k)
    print(vel)


    return B

def create_model4(n, vel, kwater):
    v1 = np.ones((n[0]))
    k = kwater
    v1[0:k] = vel[0]
    v1[k:n[0]] = np.linspace(vel[0], vel[1], n[0]-k)

    
    v1 = v1.reshape([n[0],1])

    B  = np.matlib.repmat(v1,1,n[1])
    return B






if __name__ == '__main__':

    HomeDir = './'
    RootDir = HomeDir + 'Exp1_0'
    WaveDir = HomeDir + 'Exp1_0'
    WaveDirRoot = HomeDir
    sfreq=10.0/10.0
    freq_band=np.array([0.0, 2.0])+sfreq

    if not os.path.exists(RootDir):
        os.makedirs(RootDir, exist_ok=True)
        print("Dirctory ", RootDir, " Created")

    if not os.path.exists(WaveDir):
        os.makedirs(WaveDir, exist_ok=True)
        print("Dirctory ", WaveDir, " Created")

    if not os.path.exists(WaveDirRoot):
        os.makedirs(WaveDirRoot, exist_ok=True)
        print("Dirctory ", WaveDirRoot, " Created")

    currentfile = os.path.basename(__file__)
    copy2(currentfile, RootDir)
    copy2(currentfile, WaveDir)

    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    size = comm.Get_size()
    pwrap = ParallelWrapShot()   
    # size = comm.Get_size()
    # pwrap = ParallelWrapShot()

    flag_ls = False
    flag_otl = False
    flag_ote = False
    flag_cc = False
    flag_ei = True
    flag_otq = False


    n_model3 = [101, 301]
    d_model3 = [0.03, 0.03]
    o_model3 = [0.0, 0.0]

    vtp3 = [2.0,2.3,2.6]
    vip3 = [2.0, 2.6]
    rf_loc = 81
    vt3 = create_model3(n_model3, vtp3)
    vi3 = create_model4(n_model3, vip3, 40)
    output = odn2grid(o_model3, d_model3, n_model3)
    Depth = output[0]
    Width = output[1]


    n_org = n_model3


    o_org = np.array([0.0,0.0])
    d_org = np.array(d_model3);
    model_size_phy_org = np.array([3.0, 9.0])
    n_org = np.floor(model_size_phy_org/d_org)
    n_org = n_org.astype(int) + 1
    output = odn2grid(o_org, d_org, n_org)
    Depth = output[0]
    Lateral = output[1]

    vt = vt3
    vi = vi3

    
    vt_org = copy.deepcopy(vt)
    vi_org = copy.deepcopy(vi)
    
    vt = vt.transpose()
    vi = vi.transpose()

    vt = np.reshape(vt, (np.prod(n_org), 1))
    vi = np.reshape(vi, (np.prod(n_org), 1))


    pmlx = PML(2.0, 1000)
    pmlz = PML(2.0, 1000)

    x_config = (0.0, model_size_phy_org[1], pmlx, pmlx)
    z_config = (0.0, model_size_phy_org[0], pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, n_org[1], n_org[0])
    

    C, C0, m, d = horizontal_reflector(m)

    C.data = vt
    C0.data = vi


    model_size = m._shapes[False,True]
    model_delta = m.deltas

    # xrec_multiple = np.array(range(10, model_size[0]-10)) * model_delta[0]
    # xsrc_multiple = np.linspace(xrec_multiple[0], xrec_multiple[-1], num=10)
    xsrc_multiple = np.linspace(0.3,8.7,10)
    xrec_multiple = np.linspace(0.3,8.7,281)
    print(xsrc_multiple)
    zsrc_multiple = 0.06
    zrec_multiple = 0.06
    f1 = 0.5
    f2 = 1.0
    f3 = 10.0
    f4 = 15.0 
    shotsc = equispaced_acquisition_given_locations(m,
                                                   OrmsbySource(f1,f2,f3,f4,t_shift=1.0),
                                                   sources_x_locations=xsrc_multiple,
                                                   source_depth=zsrc_multiple,
                                                   source_kwargs={},
#                                                    receivers_x_locations=[0.7, 1.7, 2.7],
                                                   receivers_x_locations=xrec_multiple,
                                                   receiver_depth=zrec_multiple,
                                                   parallel_shot_wrap=pwrap
                                                   # receiver_kwargs={'time_window':["Box", 0.9,1.5]},
                                                   )

    print(shotsc[0].sources.position)

    shots_true_multiple = copy.deepcopy(shotsc)
    shots_ini_multiple = copy.deepcopy(shotsc)
    
    # Define and configure the wave solver
    trange = (0.0,6.0)
                                   
    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=8,
                                         trange=trange,
                                         kernel_implementation='cpp',
                                         max_C=3.2)

    solver.max_C = 3.2
                                   
    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    initial_model = solver.ModelParameters(m,{'C': C0})
    print('strat generate data')

    generate_seismic_data(shots_true_multiple, solver, base_model)
#     generate_seismic_data(shotsib, solver, initial_modelb)
    generate_seismic_data(shots_ini_multiple, solver, initial_model)
                                   
    print('Data generation: {0}s'.format(time.time()-tt))

    # objective_LS = TemporalLeastSquares(solver,parallel_wrap_shot=pwrap)
    # objective_EI = TemporalEnvelope(solver, envelope_power=2.0,parallel_wrap_shot=pwrap)
    # objective_Correlate = TemporalCorrelate(solver,parallel_wrap_shot=pwrap)
    # objective_OT = TemporalOptimalTransport(solver,parallel_wrap_shot=pwrap)

    vel_bound = [1.5, 3.2]
    Boxproj = BoxConstraintPrj(vel_bound)
    PROJ_OP = Boxproj


    status_configuration = {'value_frequency'           : 1,
                                    'residual_frequency'        : 1,
                                    'residual_length_frequency' : 1,
                                    'objective_frequency'       : 1,
                                    'step_frequency'            : 1,
                                    'step_length_frequency'     : 1,
                                    'gradient_frequency'        : 1,
                                    'gradient_length_frequency' : 1,
                                    'run_time_frequency'        : 1,
                                    'alpha_frequency'           : 1,
                                    }
    line_search = 'backtrack'
    nsteps = 50
    filter_op = band_pass_filter(shots_true_multiple[0].receivers.data.shape[0], solver.tf, freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)


    if flag_ls is True:
        ExpDir = RootDir + '/LS'
        objective_LS = TemporalLeastSquares(solver,parallel_wrap_shot=pwrap, filter_op=filter_op)
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")

        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
        	os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')        
        invalg = PQN(objective_LS, proj_op=PROJ_OP)
        result_ls = invalg(shots_true_multiple, initial_model, nsteps,
                           line_search=line_search,
                           status_configuration=status_configuration, verbose=True, write=True)
        vf_ls = result_ls.C 
        vf_ls = np.reshape(vf_ls,model_size)
        vf_ls = np.transpose(vf_ls)
        vt = np.reshape(vt,model_size)
        vt = np.transpose(vt)
        vi = np.reshape(vi,model_size)
        vi = np.transpose(vi)
        if rank == 0 :
            write_data(ExpDir +  '/result.mat', vf_ls, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            write_data(ExpDir + '/objective_value_LS.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_ls.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

        comm.Barrier()

    

    if flag_otl is True:
        ExpDir = RootDir + '/OT_L'
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")

        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
        	os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')
        # objective_OT = TemporalOptimalTransport(solver,c_ratio=3.0,parallel_wrap_shot=pwrap)
        objective_OT = TemporalOptimalTransport(solver,c_ratio=5.0,parallel_wrap_shot=pwrap, filter_op=filter_op)
        invalg = PQN(objective_OT, proj_op=PROJ_OP)
        result_ot = invalg(shots_true_multiple, initial_model, nsteps,
                           line_search=line_search,
                           status_configuration=status_configuration, verbose=True, write=True)
        vf_ot = result_ot.C
        vf_ot = np.reshape(vf_ot,model_size)
        vf_ot = np.transpose(vf_ot)
        if rank == 0 :
            write_data(ExpDir +  '/result.mat', vf_ot, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            write_data(ExpDir + '/objective_value_ot_l.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_ot.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

        comm.Barrier()


    if flag_ote is True:
        ExpDir = RootDir + '/OT_E'
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")
            
        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
        	os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')
        objective_OT = TemporalOptimalTransport(solver,transform_mode='exponential',parallel_wrap_shot=pwrap,  filter_op=filter_op)
        invalg = PQN(objective_OT, proj_op=PROJ_OP)
        result_ot_e = invalg(shots_true_multiple, initial_model, nsteps,
                             line_search=line_search,
                             status_configuration=status_configuration, verbose=True, write=True)
        vf_ot_e = result_ot_e.C
        vf_ot_e = np.reshape(vf_ot_e,model_size)
        vf_ot_e = np.transpose(vf_ot_e)
        if rank == 0 :
            write_data(ExpDir +  '/result.mat', vf_ot_e, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            write_data(ExpDir + '/objective_value_ot_e.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_ot_e.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

        comm.Barrier()

    if flag_cc is True:
        ExpDir = RootDir + '/CC'
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")

        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
            os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')

        objective_Correlate = TemporalCorrelate(solver,parallel_wrap_shot=pwrap, filter_op=filter_op)
        invalg = PQN(objective_Correlate, proj_op=PROJ_OP)
        result_correlate = invalg(shots_true_multiple, initial_model, nsteps,
                                  line_search=line_search,
                                  status_configuration=status_configuration, verbose=True, write=True)
        vf_correlate = result_correlate.C
        vf_correlate = np.reshape(vf_correlate,model_size)
        vf_correlate = np.transpose(vf_correlate)
        if rank == 0 :
            write_data(ExpDir + '/result.mat', vf_correlate, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            print(len(obj_vals_i))
            write_data(ExpDir + '/objective_value_correlate.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_correlate.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

        comm.Barrier()
        

    if flag_ei is True:    
        ExpDir = RootDir + '/EIP1'
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")

        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
            os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')
        objective_EI = TemporalEnvelope(solver, envelope_power=1.0,parallel_wrap_shot=pwrap, filter_op=filter_op)
        invalg = PQN(objective_EI, proj_op=PROJ_OP)
        result_ei = invalg(shots_true_multiple, initial_model, nsteps,
                           line_search=line_search,
                           status_configuration=status_configuration, verbose=True, write=True)
        vf_ei = result_ei.C
        vf_ei = np.reshape(vf_ei,model_size)
        vf_ei = np.transpose(vf_ei)
        if rank == 0 :
            write_data(ExpDir + '/result.mat', vf_ei, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            write_data(ExpDir + '/objective_value_ei.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_ei.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)
        
        comm.Barrier()

    if flag_otq is True:
        ExpDir = RootDir + '/OT_Q'
        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")

        os.chdir(ExpDir)
        if not os.path.exists(ExpDir+'/xm'):
            os.makedirs(ExpDir+'/xm', exist_ok=True)
            
        os.chdir(ExpDir+'/xm')
        objective_OT = TemporalOptimalTransport(solver,transform_mode='quadratic',parallel_wrap_shot=pwrap, filter_op=filter_op)
        invalg = PQN(objective_OT, proj_op=PROJ_OP)
        result_ot_q = invalg(shots_true_multiple, initial_model, nsteps,
        	                 line_search=line_search,
              	             status_configuration=status_configuration, verbose=True, write=True)
        vf_ot_q = result_ot_q.C
        vf_ot_q = np.reshape(vf_ot_q,model_size)
        vf_ot_q = np.transpose(vf_ot_q)
        if rank == 0 :
            write_data(ExpDir +  '/result.mat', vf_ot_q, [0.0,0.0], d_org, n_org)
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            write_data(ExpDir + '/objective_value_ot_q.mat', obj_vals_i, 0, 1, len(obj_vals_i))

            write_data(ExpDir +  '/true.mat', vt, [0.0,0.0], d_org, n_org)
            write_data(ExpDir +  '/initial.mat', vi, [0.0,0.0], d_org, n_org)
            dobs = shots_true_multiple[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots_true_multiple[0].receivers.receiver_list[1].position[0]-shots_true_multiple[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots_true_multiple[0].receivers.receiver_list[0].position[0]]
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            shots_final = copy.deepcopy(shots_true_multiple)
            final_model = solver.ModelParameters(m,{'C': result_ot_q.C})
            generate_seismic_data(shots_final, solver, final_model)
            dpred = shots_final[0].receivers.data
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

        comm.Barrier()

    cmd = 'cp -r '+RootDir+' ' + WaveDirRoot
    os.system(cmd)

    




