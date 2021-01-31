

import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.util.compute_tools import correlate_fun
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit.util.wave_compress import *

__all__ = ['TemporalCorrelate']

__docformat__ = "restructuredtext en"

class TemporalCorrelate(ObjectiveFunctionBase):
    """ How to compute the parts of the objective you need to do optimization """

    def __init__(self, solver, filter_op=None, zero_lag=False, WaveCompressInfo=None, parallel_wrap_shot=ParallelWrapShotNull(), imaging_period=1):
        """imaging_period: Imaging happens every 'imaging_period' timesteps. Use higher numbers to reduce memory consumption at the cost of lower gradient accuracy.
            By assigning this value to the class, it will automatically be used when the gradient function of the temporal objective function is called in an inversion context.
        """
        self.solver = solver
        self.WaveCompressInfo = WaveCompressInfo
        self.modeling_tools = TemporalModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot

        self.imaging_period = int(imaging_period) #Needs to be an integer
        self.filter_op = filter_op
        self.zero_lag = zero_lag

    def _residual(self, shot, m0, dWaveOp=None, wavefield=None):
        """Computes residual in the usual sense.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.
        dWaveOp : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """

        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        rp = ['simdata']
        if dWaveOp is not None:
            rp.append('dWaveOp')
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')

        # Run the forward modeling step
        retval = self.modeling_tools.forward_model(shot, m0, self.imaging_period, return_parameters=rp,dWaveOp=dWaveOp)

        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
        # resid = shot.receivers.interpolate_data(self.solver.ts()) - retval['simdata']
        
        if shot.background_data is not None:
            dpred = retval['simdata'] - shot.background_data
        else:
            dpred = retval['simdata']

        if shot.receivers.time_window is None:
            # dpred = retval['simdata']
            dpred = dpred
        else:
            # dpred = shot.receivers.time_window(self.solver.ts()) * retval['simdata']
            dpred = shot.receivers.time_window(self.solver.ts()) * dpred

        if self.filter_op is not None:
            dobs = self.filter_op * shot.receivers.interpolate_data(self.solver.ts())
            dpred = self.filter_op * dpred
            # resid = self.filter_op * resid
            # adjoint_src = self.filter_op * resid
        else:
            dobs = shot.receivers.interpolate_data(self.solver.ts())
            # dpred = retval['simdata']
            # adjoint_src = resid

        # n_correlate_data = shape_dobs[0]
        shape_dobs = np.shape(dobs)
        # n_correlate_data = shape_dobs[0]*2-1
        n_correlate_data = shape_dobs[0]
        resid = np.zeros([n_correlate_data, shape_dobs[1]])
        adjoint_src = np.zeros([shape_dobs[0], shape_dobs[1]])
        # W = np.linspace(-self.solver.dt*(shape_dobs[0]-1), self.solver.dt*(shape_dobs[0]-1), shape_dobs[0]*2-1)
        # W1 = np.linspace(-self.solver.dt*(shape_dobs[0]-1)/2.0, self.solver.dt*(shape_dobs[0]-1)/2.0, n_correlate_data)
        if self.zero_lag is False:
            W  = np.zeros(n_correlate_data)
            if np.mod(n_correlate_data, 2) == 0:
                W[0:n_correlate_data//2] = np.linspace(-self.solver.dt, -self.solver.dt*(shape_dobs[0])/2.0, n_correlate_data/2)
                W[n_correlate_data//2:n_correlate_data] = np.flipud(W[0:n_correlate_data//2])
            else:
                W[0:(n_correlate_data+1)//2] = np.linspace(0.0, -self.solver.dt*(shape_dobs[0]-1)/2.0, (n_correlate_data+1)/2)
                W[(n_correlate_data-1)//2:n_correlate_data] = np.flipud(W[0:(n_correlate_data+1)//2])

            # W[0: (n_correlate_data+1)//2] = W1[(n_correlate_data+1)//2-1:n_correlate_data]
            # W[(n_correlate_data+1)//2:n_correlate_data] = W1[0:n_correlate_data+1)//2]
            # W = np.linspace(0.0, self.solver.dt*(shape_dobs[0]-1), n_correlate_data)
            W = np.abs(W)
            W = np.sqrt(W)
            for i in range(0, shape_dobs[1]):
                correlate_data = correlate_fun(dobs[:,i], dpred[:,i])
                Wf = W * correlate_data
                correlate_data_norm2 = np.dot(correlate_data, correlate_data)
                Wf_norm2 = np.dot(Wf, Wf)
                resid[:, i] = Wf / np.sqrt(correlate_data_norm2)
                
                adjoint_src[:,i] = (2.0*correlate_data_norm2*correlate_fun(dobs[:,i], W*Wf, mode='adj') - 2.0*Wf_norm2*correlate_fun(dobs[:,i], correlate_data, mode='adj')) / correlate_data_norm2**2.0

        else:
            resid = np.zeros([1, shape_dobs[1]])
            for i in range(0, shape_dobs[1]):
                dobsi = dobs[:,i]
                dpredi = dpred[:,i]
                norm_dobsi = np.linalg.norm(dobsi)
                norm_dpredi = np.linalg.norm(dpredi)
                dobsi_n = dobsi / norm_dobsi
                dpredi_n = dpredi / norm_dpredi
                resid[:,i] = -np.sum(dobsi_n * dpredi_n)
                adjoint_src_1 = - dobsi / norm_dpredi / norm_dobsi
                adjoint_src_2 = -resid[:,i] * dpredi / norm_dpredi**2.0
                adjoint_src[:,i] = adjoint_src_1 + adjoint_src_2

        if self.filter_op is not None:
            adjoint_src = self.filter_op.__adj_mul__(adjoint_src)


        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            # dWaveOp[:] = retval['dWaveOp'][:]
            dWaveOp = retval['dWaveOp']
        if wavefield is not None:
            wavefield[:] = retval['wavefield'][:]

        return resid, -1*adjoint_src

    def evaluate(self, shots, m0, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        r_norm2 = 0
        for shot in shots:
            r, adjoint_src = self._residual(shot, m0)
            if self.zero_lag is False:
                r_norm2 += np.linalg.norm(r)**2
            else:
                r_norm2 += np.sum(r)

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

        return r_norm2*self.solver.dt

    def _gradient_helper(self, shot, m0, ignore_minus=False, ret_pseudo_hess_diag_comp = False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """

        # Compute the residual vector and its norm
        if self.WaveCompressInfo is None:
            dWaveOp=[]
        else:
            WaveCompressInfoLocal = copy.deepcopy(self.WaveCompressInfo)
            dWaveOp=CompressWaveList(WaveCompressInfoLocal)

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0,'rho'):
            wavefield=[]
        else:
            wavefield=None
            
        r, adjoint_src = self._residual(shot, m0, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)
        
        # Perform the migration or F* operation to get the gradient component
        g = self.modeling_tools.migrate_shot(shot, m0, adjoint_src, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)

        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return g, r, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            return g, r

    def _pseudo_hessian_diagonal_component_shot(self, dWaveOp):
        #Shin 2001: "Improved amplitude preservation for prestack depth migration by inverse scattering theory". 
        #Basic illumination compensation. In here we compute the diagonal. It is not perfect, it does not include receiver coverage for instance.
        #Currently only implemented for temporal modeling. Although very easy for frequency modeling as well. -> np.real(omega^4*wavefield * np.conj(wavefield)) -> np.real(dWaveOp*np.conj(dWaveOp))
        
        mesh = self.solver.mesh
          
        import time
        tt = time.time()
        pseudo_hessian_diag_contrib = np.zeros(mesh.unpad_array(dWaveOp[0], copy=True).shape)
        for i in range(len(dWaveOp)):                          #Since dWaveOp is a list I cannot use a single numpy command but I need to loop over timesteps. May have been nicer if dWaveOp had been implemented as a single large ndarray I think
            unpadded_dWaveOp_i = mesh.unpad_array(dWaveOp[i])   #This will modify dWaveOp[i] ! But that should be okay as it will not be used anymore.
            pseudo_hessian_diag_contrib += unpadded_dWaveOp_i*unpadded_dWaveOp_i

        pseudo_hessian_diag_contrib *= self.imaging_period #Compensate for doing fewer summations at higher imaging_period

        print("Time elapsed when computing pseudo hessian diagonal contribution shot: %e"%(time.time() - tt))

        return pseudo_hessian_diag_contrib

    def compute_gradient(self, shots, m0, aux_info={}, **kwargs):
        """Compute the gradient for a set of shots.

        Computes the gradient as
            -F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the gradient.
        m0 : ModelParameters
            The base point about which to compute the gradient
        """


        # compute the portion of the gradient due to each shot
        grad = m0.perturbation()
        r_norm2 = 0.0
        pseudo_h_diag = np.zeros(m0.asarray().shape)
        for shot in shots:
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                g, r, h = self._gradient_helper(shot, m0, ignore_minus=True, ret_pseudo_hess_diag_comp = True, **kwargs)
                pseudo_h_diag += h 
            else:
                g, r = self._gradient_helper(shot, m0, ignore_minus=True, **kwargs)
            
            grad -= g # handle the minus 1 in the definition of the gradient of this objective
            if self.zero_lag is False:
                r_norm2 += np.linalg.norm(r)**2
            else:
                r_norm2 += np.sum(r)

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

            ngrad = np.zeros_like(grad.asarray())
            self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
            grad=m0.perturbation(data=ngrad)
            
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                pseudo_h_diag_temp = np.zeros(pseudo_h_diag.shape)
                self.parallel_wrap_shot.comm.Allreduce(pseudo_h_diag, pseudo_h_diag_temp)
                pseudo_h_diag = pseudo_h_diag_temp 

        # account for the measure in the integral over time
        r_norm2 *= self.solver.dt
        pseudo_h_diag *= self.solver.dt #The gradient is implemented as a time integral in TemporalModeling.adjoint_model(). I think the pseudo Hessian (F*F in notation Shin) also represents a time integral. So multiply with dt as well to be consistent.

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, r_norm2)
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            aux_info['pseudo_hess_diag'] = (True, pseudo_h_diag)

        return grad

    def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        result = m0.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:
                # Run the forward modeling step
                retval = self.modeling_tools.forward_model(shot, m0, return_parameters=['dWaveOp'])
                dWaveOp0 = retval['dWaveOp']

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata'], dWaveOp0=dWaveOp0)

                d1 = linear_retval['simdata'] # data from F applied to m1
                result += self.modeling_tools.migrate_shot(shot, m0, d1, dWaveOp=dWaveOp0)

        elif hessian_mode == 'full':
            for shot in shots:
                # Run the forward modeling step
                dWaveOp0 = list() # wave operator derivative wrt model for u_0
                r0, adjoint_src = self._residual(shot, m0, dWaveOp=dWaveOp0, **kwargs)

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
                d1 = linear_retval['simdata']
                dWaveOp1 = linear_retval['dWaveOp1']

                # <q, u1tt>, first adjointy bit
                dWaveOpAdj1=[]
                res1 = self.modeling_tools.migrate_shot( shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
                result += res1

                # <p, u0tt>
                res2 = self.modeling_tools.migrate_shot(shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
                result += res2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:

            nresult = np.zeros_like(result.asarray())
            self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
            result = m0.perturbation(data=nresult)

        # Note, AFTER the application has been done in parallel do this.
        if hessian_mode == 'levenberg':
            result += levenberg_mu*m1

        return result
