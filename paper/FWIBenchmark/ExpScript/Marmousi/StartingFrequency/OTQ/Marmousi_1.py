# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import math
import os
import scipy.io as sio
from shutil import copy2

import sys

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.gallery.layered_medium import three_layered_medium
from pysit.gallery import marmousi
from pysit.gallery import marmousi2
from pysit.util.io import *
from pysit.util.compute_tools import *

from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup

    noise_percentage = 0.0 # the ratior between noise and signal
    nsteps = 50 # number of iteration per frquency band
    sfreq=16.0
    sfreq=sfreq/10.0
    freq_bands = [[sfreq,sfreq+3.0],[sfreq,40.5]] # frequency bands used in the inversion

    ALPHA_ALL = [0.0, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    

    EXP_DIR_STR = ['_alpha_00', '_alpha_01', '_alpha_02', '_alpha_03', '_alpha_04',
                   '_alpha_05', '_alpha_06', '_alpha_07', '_alpha_08', '_alpha_09', '_alpha_10',]

    Nshots = 20
    jobs_file = '/scratch/slim/myang/Zhilong/job_history.txt'

    for exp_ind in range(len(ALPHA_ALL)-1, len(ALPHA_ALL)):
        # freq_bands = [[2.5, 40.5]]
        exp_dir_str_i = EXP_DIR_STR[exp_ind]

        # RootDir = '/Users/fangzl/Data/Result'
        HomeDir = './'
        RootDir = HomeDir
        WaveRootDir = HomeDir
        # SubDir = '/OT_Hard_PML50_Initial00_4s_3_7Hz'
        SubDir = '/OTQ_BoxConstraint_Starting2_1_4_1Hz_CorrectBox_Wolfe'

        ExpDir = RootDir + SubDir
        # ExpDir = './Model'
        WaveDir = WaveRootDir + SubDir
        ExpDir = WaveDir
        
        alpha = ALPHA_ALL[exp_ind]

        if not os.path.exists(ExpDir):
            os.makedirs(ExpDir, exist_ok=True)
            print("Dirctory ", ExpDir, " Created")
        
        if not os.path.exists(WaveDir):
            os.makedirs(WaveDir, exist_ok=True)
            print("Dirctory ", WaveDir, " Created")

        currentfile = os.path.basename(__file__)
        copy2(currentfile, ExpDir)
        copy2(currentfile, WaveDir)

        logfile=ExpDir + '/exp.out'
        fptlogfile=open(logfile,'w+')

        os.chdir(ExpDir)

        comm = MPI.COMM_WORLD
    #   comm = MPI.COMM_SELF
        size = comm.Get_size()
        rank = comm.Get_rank()

        # if comm.Get_rank() == 0:
        #     fpt2=open(jobs_file,'a+')
        #     fpt2.write("EXP Dir            :" + ExpDir + "\n")
        #     fpt2.close()

        pwrap = ParallelWrapShot()

        if rank == 0:
            ttt = time.time()

        # #   Define Domain
        # pmlx = PML(0.1, 100)
        # pmlz = PML(0.1, 100)

        C, C1, m, d = marmousi(patch='mini_square')
        # C, C2, m, d = marmousi(patch='mini_square',initial_model_style='gradient', initial_config={'min':1500.0, 'max':3500.0})
        C, C2, m, d = marmousi(patch='mini_square',initial_model_style='smooth_low_pass', initial_config={'freq':1/1500.})
        C0 = (1-alpha)*C1 + alpha*C2
        C = C / 1000.0
        C0 = C0 / 1000.0
        m_shape = m._shapes[(False,True)]
        d_model = m.deltas
        pmlx = PML(1.0, 1500)
        pmlz = PML(1.0, 1500)
        x_config = (d.x.lbound/1000.0, d.x.rbound/1000.0, pmlx, pmlx)
        z_config = (d.z.lbound/1000.0, d.z.rbound/1000.0, pmlz, pmlz)
        d = RectangularDomain(x_config, z_config)
        m = CartesianMesh(d, m_shape[0], m_shape[1])

        # C, C1, m, d = three_layered_medium(TrueModelFileName='testtrue.mat',           InitialModelFileName='testInitial.mat',
        #                                    initial_model_style='gradient',
        #                                    initial_config={'sigma': 4.0, 'filtersize': 4},
        #                                    vels=(1.5,2.0,2.5),
        #                                    pml_width=[0.3,0.3],
        #                                    water_layer_depth=0.05)

        # C, C2, m, d = three_layered_medium(TrueModelFileName='testtrue.mat',           InitialModelFileName='testInitial.mat',
        #                                    initial_model_style='smooth',
        #                                    initial_config={'sigma': 6.0, 'filtersize': 4},
        #                                    vels=(1.5,2.0,2.5),
        #                                    pml_width=[0.3,0.3],
        #                                    water_layer_depth=0.05)
        # # quit()

        # C0 = alpha * C2 + (1-alpha) * C1
        # C, C0, m, d = three_layered_medium()

        # x_config = (0.1, 1.0, pmlx, pmlx)
        # z_config = (0.1, 0.8, pmlz, pmlz)

        # d = RectangularDomain(x_config, z_config)

        # m = CartesianMesh(d, 91, 71)

        # #   Generate true wave speed
        # C, C0, m, d = horizontal_reflector(m)

        # Set up shots
        zmin = d.z.lbound
        zmax = d.z.rbound
        # zpos = zmin + (1./9.)*zmax
        zpos = 0.01 * 2.0
        f1 = 0.5
        f2 = 1.0
        f3 = 10.0
        f4 = 15.0

        
        sys.stdout.write("{0}: {1}\n".format(rank, Nshots / size))

        shots = equispaced_acquisition(m,
                                       OrmsbySource(f1,f2,f3,f4,t_shift=2.0),
                                       sources=Nshots,
                                       source_depth=zpos,
                                       source_kwargs={},
                                       receivers='max',
                                       receiver_depth=zpos,
                                       receiver_kwargs={},
                                       parallel_shot_wrap=pwrap,
                                       )


        # Define and configure the wave solver
        trange = (0.0,8.0)

        solver = ConstantDensityAcousticWave(m,
                                             spatial_accuracy_order=8,
                                             trange=trange,
                                             kernel_implementation='cpp',
                                             max_C=5.5) # The dt is automatically fixed for given max_C (velocity)
        solver.max_C = 5.5

        # Generate synthetic Seismic data
        sys.stdout.write('Generating data...')
        base_model = solver.ModelParameters(m,{'C': C})
        initial_model = solver.ModelParameters(m,{'C': C0})
        tt = time.time()
        generate_seismic_data(shots, solver, base_model)
        shots_ini = copy.deepcopy(shots)
        generate_seismic_data(shots_ini, solver, initial_model)
        sys.stdout.write('{1}:Data generation: {0}s\n'.format(time.time()-tt,rank))

        sys.stdout.flush()

        for i in range(len(shots)):
            data_norm = np.linalg.norm(shots[i].receivers.data.flatten())
            noise = np.random.normal(0.0, 1.0, shots[i].receivers.data.shape)
            noise_norm = np.linalg.norm(noise.flatten())
            noise = noise / noise_norm * data_norm * noise_percentage
            shots[i].receivers.data += noise


        comm.Barrier()

        if rank == 0:
            tttt = time.time()-ttt
            sys.stdout.write('Total wall time: {0}\n'.format(tttt))
            sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

        

        obj_vals_list = []    
        n_iter_used = np.zeros((len(freq_bands),1), dtype=int)
        initial_value = solver.ModelParameters(m, {'C': C0})
        initial_value.data = initial_value.with_padding(padding_mode='edge').data
        initial_value.padded = True

        for i_freq in range(len(freq_bands)):
            dir_ifreq = 'xm'+str(i_freq+1)
            
            if rank == 0:
                if not os.path.exists(dir_ifreq):
                    os.mkdir(dir_ifreq)
            comm.Barrier()
            os.chdir(dir_ifreq)

            freq_band = freq_bands[i_freq]
            filter_op = band_pass_filter(shots[0].receivers.data.shape[0], solver.tf, freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)
            # filter_op = None

            c_shots = np.zeros(len(shots))
            for i_shot in range(len(shots)):
                dobs_ishot = shots[i_shot].receivers.data
                dpred_ishot = shots_ini[i_shot].receivers.data
                shape_dobs = np.shape(dobs_ishot)
                c_ratio_irec = np.zeros(shape_dobs[1])

                if filter_op is not None:
                    dobs_ishot = filter_op * dobs_ishot
                    dpred_ishot = filter_op * dpred_ishot

                for i_rec in range(0, shape_dobs[1]):
                    dobs_irec = dobs_ishot[:,i_rec]
                    dpred_irec = dpred_ishot[:,i_rec]

                    dobs_max = np.max(np.abs(dobs_irec))
                    dpred_min_abs = np.abs(np.min(dpred_irec))

                    c_ratio_irec[i_rec] = np.max((1.0, dpred_min_abs/dobs_max))

                c_shots[i_shot] = np.max(c_ratio_irec)


            c_ratio_local = np.max(c_shots)
            c_ratio = c_ratio_local * 1.5

            # objective = TemporalEnvelope(solver, envelope_power=2.0, filter_op=filter_op, parallel_wrap_shot=pwrap)
            if i_freq == 0:
                objective = TemporalOptimalTransport(solver, filter_op=filter_op, parallel_wrap_shot=pwrap, transform_mode='quadratic', c_ratio=c_ratio)
            else:
                # objective = TemporalOptimalTransport(solver, filter_op=filter_op, parallel_wrap_shot=pwrap, transform_mode='quadratic', c_ratio=c_ratio)
            # objective = TemporalOptimalTransport(solver, filter_op=filter_op, parallel_wrap_shot=pwrap, transform_mode='exponential', exp_a=1.0)
                objective = TemporalLeastSquares(solver, filter_op=filter_op, parallel_wrap_shot=pwrap)

            # Define the inversion algorithm
            # invalg = GradientDescent(objective)
            vel_bound = [1.5, 5.5]
            PMLproj = PMLExtensionPrj()
            Boxproj = BoxConstraintPrj(vel_bound)
            
            ModelSize = initial_value.mesh._shapes[(True,True)]
            NumWaterLayer = 28
            WaterVel = 1.5
            WaterPrj = WaterVelocityPrj(ModelSize, NumOfWaterLayer=NumWaterLayer, WaterVel=WaterVel)
            Proj_Op  = JointPrj(WaterPrj, PMLproj)
           # Proj_Op = JointPrj(Proj_Op1, Boxproj)
            # Proj_Op = JointPrj(PMLproj, Boxproj)
            # Proj_Op = PMLproj
            Proj_Op = Boxproj

            invalg = PQN(objective, proj_op=Proj_Op, memory_length=10, maxiter_PGD=1000, PQN_start_iteration=3)
            invalg.logfile = fptlogfile
            

            # Execute inversion algorithm
            print('Running LBFGS...')
            tt = time.time()

            

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


            # line_search = 'backtrack'
            line_search = 'Wolfe'

            result = invalg(shots, initial_value, nsteps,
                            line_search=line_search,
                            tolerance=1e-11,
                            status_configuration=status_configuration, verbose=True, write=True)

            print('Run time:  {0}s'.format(time.time()-tt))

            initial_value.data = result.C
            obj_vals_i = np.array([v for k,v in list(invalg.objective_history.items())])
            obj_vals_list.append(obj_vals_i)
            n_iter_used[i_freq] = len(obj_vals_i)
            write_data('./objective_value.mat', obj_vals_i, 0, 1, len(obj_vals_i))
            os.chdir('../')

        



        C_cut = initial_value.without_padding().data
        # obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])
        comm.Barrier()
        if rank == 0:

            obj_vals = np.zeros((np.max(n_iter_used), len(freq_bands)))

            for i in range(len(freq_bands)):
                obj_vals[0:int(n_iter_used[i]), i] = obj_vals_list[i]

            # model = result.C.reshape(m.shape(as_grid=True)).transpose()
            model = C_cut.reshape(m.shape(as_grid=True)).transpose()

            nt = m.shape(as_grid=True)
            nt = (nt[1], nt[0])
            dt = (m.parameters[1].delta, m.parameters[0].delta)
            ot = (d.z.lbound, d.x.lbound)
            
            

            write_data(ExpDir + '/result.mat', model, ot, dt, nt)
            write_data(WaveDir + '/result.mat', model, ot, dt, nt)
            write_data(ExpDir + '/true.mat', C.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
            write_data(WaveDir + '/true.mat', C.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
            write_data(ExpDir + '/initial.mat', C0.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
            write_data(WaveDir + '/initial.mat', C0.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
            write_data(ExpDir + '/objective_value.mat', obj_vals, [0, 0], [1, 1], np.shape(obj_vals))
            write_data(WaveDir + '/objective_value.mat', obj_vals, [0, 0], [1, 1], np.shape(obj_vals))

            n_wPMLo = initial_value.mesh._shapes[(True,True)]
            n_wPML = [n_wPMLo[1], n_wPMLo[0]]
            n_fband = len(freq_bands)
            
            vi, ot, dt, nt = read_data('initial.mat')
            vt, ot, dt, nt = read_data('true.mat')
            vi = vi.flatten()
            vt = vt.flatten()

            dv0 = np.linalg.norm(vt-vi)
            m_err = [1.0]
        
            for i_freq in range(n_fband):
                dir = './xm' + str(i_freq+1)
            
                for k in range(nsteps):
                    file = dir+'/x_'+str(k+1)+'.mat'
                    A = sio.loadmat(file)
                    vi = A['data']
                    vi = np.reshape(vi, n_wPML, order='F')
                    npml = (n_wPML - nt) // 2
                    indx_1 = [npml[0], npml[0]+nt[0]]
                    indx_2 = [npml[1], npml[1]+nt[1]]
                    vk = vi[indx_1[0]:indx_1[1], indx_2[0]:indx_2[1]]
                    vk = vk.flatten()
                    m_err.append(np.linalg.norm(vk-vt)/dv0)

            write_data(ExpDir + '/model_error.mat',np.array(m_err), 0,1,len(m_err))
            write_data(WaveDir + '/model_error.mat',np.array(m_err), 0,1,len(m_err))

            print('write data')

            dobs = shots[0].receivers.data 
            ndata = np.shape(dobs)
            ddata = [shots[0].receivers.receiver_list[1].position[0]-shots[0].receivers.receiver_list[0].position[0], solver.dt]
            odata = [0.0, shots[0].receivers.receiver_list[0].position[0]]
            write_data(WaveDir + '/Dobs.mat', dobs, odata, ddata, ndata)
            write_data(ExpDir + '/Dobs.mat', dobs, odata, ddata, ndata)

            final_model = solver.ModelParameters(m,{'C': C_cut})
            generate_seismic_data(shots, solver, final_model)
            dpred = shots[0].receivers.data
            write_data(WaveDir + '/Dpred.mat', dpred, odata, ddata, ndata)
            write_data(ExpDir + '/Dpred.mat', dpred, odata, ddata, ndata)

            initial_model = solver.ModelParameters(m,{'C': C0})
            generate_seismic_data(shots, solver, initial_model)
            dpred = shots[0].receivers.data
            write_data(WaveDir + '/Dini.mat', dpred, odata, ddata, ndata)
            write_data(ExpDir + '/Dini.mat', dpred, odata, ddata, ndata)
            
            print('write data finish')

            logfile=RootDir + '/exp.out'
            copy2(logfile, ExpDir)
            copy2(logfile, WaveDir)







        # Do something to visualize the results
    #   display_on_grid(C, d, shade_pml=True)
    #   display_on_grid(result.C, d, shade_pml=True)
        #display_seismogram(shots[0], clim=[-1,1])
        #display_seismogram(shots[0], wiggle=True, wiggle_skip=1)
        # animate_wave_evolution(ps, domain=d, display_rate=10, shade_pml=True)
