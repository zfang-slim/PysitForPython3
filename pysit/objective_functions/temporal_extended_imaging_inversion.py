

import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit.optimization.extended_least_squares_migration import ExtendLSM
from pysit.util.wave_compress import *

__all__ = ['TemporalExtendedImagingInversion']

__docformat__ = "restructuredtext en"


class TemporalExtendedImagingInversion(ObjectiveFunctionBase):
    """ How to compute the parts of the objective you need to do optimization """

    def __init__(self, solver, h, filter_op=None, dm_extend=None, max_sub_offset=0.0,
                 weight_matrix=None, regularization_value=None, krylov_maxiter=10,
                 parallel_wrap_shot=ParallelWrapShotNull(), imaging_period=1):
        """imaging_period: Imaging happens every 'imaging_period' timesteps. Use higher numbers to reduce memory consumption at the cost of lower gradient accuracy.
            By assigning this value to the class, it will automatically be used when the gradient function of the temporal objective function is called in an inversion context.
        """
        self.solver = solver
        self.modeling_tools = TemporalModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot

        self.imaging_period = int(imaging_period)  # Needs to be an integer
        self.filter_op = filter_op

        self.weight_matrix = weight_matrix
        self.regularization_value = regularization_value
        self.krylov_maxiter = krylov_maxiter

        self.max_sub_offset = max_sub_offset
        self.h = h
        self.dm_extend = dm_extend

    def evaluate(self, shots, m0, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        lindatas = dict()
        for i in range(len(shots)):
            lindatas[i] = dict()
            lindatas[i] = shots[i].receivers.data

        LSM_obj = ExtendLSM(self.modeling_tools, shots, m0, lindatas, self.max_sub_offset, self.h,
                            imaging_period=1, krylov_maxiter=self.krylov_maxiter, 
                            weight_matrix=self.weight_matrix, 
                            regularization_value=self.regularization_value, 
                            parallel_wrap_shot=self.parallel_wrap_shot)
        if self.dm_extend is not None:
            dm_extend = LSM_obj.run_lsm(self.dm_extend)
        else:
            dm_extend = LSM_obj.run_lsm()


        r_norm2 = 0
        for shot in shots:
            r, adjoint_src = self._residual(shot, m0, dm_extend)
            r_norm2 += np.linalg.norm(r)**2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(
                np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()]  # goofy way to access 0-D array element

        return 0.5*r_norm2*self.solver.dt + self._regularization_fun(m0, dm_extend)*self.solver.dt

    def _regularization_fun(self, m0, dm_extend):
        if self.regularization_value is None:
            reg_value = 0.0 
        else:
            sh_data = dm_extend.sh_data
            max_sub_offset = dm_extend.max_sub_offset
            if self.weight_matrix == 'linear_h':
                weight_h = np.linspace(-max_sub_offset,
                                       max_sub_offset, sh_data[1])
                reg_value = 0.0
                for i in range(sh_data[1]):
                    f_dm_i = weight_h[i] * dm_extend.data[:, i] 
                    reg_value += 0.5*np.linalg.norm(f_dm_i)**2.0 * self.regularization_value

                reg_value *= np.prod(m0.mesh.deltas)

        return reg_value

    def _residual(self, shot, m0, dm_extend, dWaveOp0=None, dWaveOp1=None, wavefield=None):
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
        if dWaveOp1 is not None:
            rp.append('dWaveOp1')
        if dWaveOp0 is not None:
            rp.append('dWaveOp0')
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')

        # Run the forward modeling step
        retval = self.modeling_tools.linear_forward_model_extend([shot], m0, dm_extend, self.max_sub_offset,
                                                                 self.h, return_parameters=rp)

        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
        
        if shot.background_data is not None:
            dpred = retval['simdata'][0] - shot.background_data
        else:
            dpred = retval['simdata'][0]

        if shot.receivers.time_window is None:
            # dpred = retval['simdata']
            dpred = dpred
            resid = shot.receivers.interpolate_data(self.solver.ts()) - dpred
        else:
            # dpred = shot.receivers.time_window(self.solver.ts()) * retval['simdata']
            dpred = shot.receivers.time_window(self.solver.ts()) * dpred
            resid = shot.receivers.interpolate_data(self.solver.ts()) - dpred
        
        # if shot.receivers.time_window is None:
        #     resid = shot.receivers.interpolate_data(self.solver.ts()) - retval['simdata'][0]
        # else:
        #     dpred = shot.receivers.time_window(self.solver.ts()) * retval['simdata'][0]
        #     resid = shot.receivers.interpolate_data(self.solver.ts()) - dpred

        # resid = shot.receivers.interpolate_data(self.solver.ts()) - retval['simdata'][0]
        if self.filter_op is not None:
            resid = self.filter_op * resid
            adjoint_src = self.filter_op.__adj_mul__(resid)
        else:
            adjoint_src = resid

        # If the second derivative info is needed, copy it out
        if dWaveOp0 is not None:
            dWaveOp0[:] = retval['dWaveOp0'][0][:]
        if dWaveOp1 is not None:
            dWaveOp1[:] = retval['dWaveOp1'][0][:]
        if wavefield is not None:
            wavefield[:] = retval['wavefield'][0][:]

        return resid, adjoint_src

    def _gradient_helper(self, shot, m0, dm_extend, ignore_minus=False, ret_pseudo_hess_diag_comp=False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """

        # Compute the residual vector and its norm
        dWaveOp0 = []
        dWaveOp1 = []

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0, 'rho'):
            wavefield = []
        else:
            wavefield = None

        r, adjoint_src = self._residual(
            shot, m0, dm_extend, dWaveOp0=dWaveOp0, dWaveOp1=dWaveOp1, wavefield=wavefield, **kwargs)

        # Perform the migration or F* operation to get the gradient component
        # g = self.modeling_tools.migrate_shot(shot, m0, adjoint_src, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)
        g = self.modeling_tools.gradient_extend_shot(shot, m0, dm_extend, adjoint_src, self.max_sub_offset, self.h, self.imaging_period,
                                                     dWaveOp0=dWaveOp0, dWaveOp1=dWaveOp1, wavefield=wavefield)

        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return g, r, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            return g, r

    def _pseudo_hessian_diagonal_component_shot(self, dWaveOp):
        # Shin 2001: "Improved amplitude preservation for prestack depth migration by inverse scattering theory".
        # Basic illumination compensation. In here we compute the diagonal. It is not perfect, it does not include receiver coverage for instance.
        # Currently only implemented for temporal modeling. Although very easy for frequency modeling as well. -> np.real(omega^4*wavefield * np.conj(wavefield)) -> np.real(dWaveOp*np.conj(dWaveOp))

        mesh = self.solver.mesh

        import time
        tt = time.time()
        pseudo_hessian_diag_contrib = np.zeros(mesh.unpad_array(dWaveOp[0], copy=True).shape)
        for i in range(len(dWaveOp)):  # Since dWaveOp is a list I cannot use a single numpy command but I need to loop over timesteps. May have been nicer if dWaveOp had been implemented as a single large ndarray I think
            # This will modify dWaveOp[i] ! But that should be okay as it will not be used anymore.
            unpadded_dWaveOp_i = mesh.unpad_array(dWaveOp[i])
            pseudo_hessian_diag_contrib += unpadded_dWaveOp_i*unpadded_dWaveOp_i

        # Compensate for doing fewer summations at higher imaging_period
        pseudo_hessian_diag_contrib *= self.imaging_period

        print("Time elapsed when computing pseudo hessian diagonal contribution shot: %e" %
              (time.time() - tt))

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

        lindatas = dict()
        for i in range(len(shots)):
            lindatas[i] = dict()
            lindatas[i] = shots[i].receivers.data

        LSM_obj = ExtendLSM(self.modeling_tools, shots, m0, lindatas, self.max_sub_offset, self.h,
                            imaging_period=self.imaging_period, krylov_maxiter=self.krylov_maxiter, 
                            regularization_value=self.regularization_value,
                            weight_matrix=self.weight_matrix,
                            parallel_wrap_shot=self.parallel_wrap_shot)
        if self.dm_extend is not None:
            dm_extend = LSM_obj.run_lsm(self.dm_extend)
        else:
            dm_extend = LSM_obj.run_lsm()
        # self.dm_extend = dm_extend

        grad = m0.perturbation()
        r_norm2 = 0.0
        pseudo_h_diag = np.zeros(m0.asarray().shape)
        for shot in shots:
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                g, r, h = self._gradient_helper(shot, m0, ignore_minus=True, ret_pseudo_hess_diag_comp=True, **kwargs)
                pseudo_h_diag += h
            else:
                g, r = self._gradient_helper(shot, m0, dm_extend, ignore_minus=True, **kwargs)

            grad -= g  # handle the minus 1 in the definition of the gradient of this objective
            r_norm2 += np.linalg.norm(r)**2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()]  # goofy way to access 0-D array element

            ngrad = np.zeros_like(grad.asarray())
            self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
            grad = m0.perturbation(data=ngrad)

            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                pseudo_h_diag_temp = np.zeros(pseudo_h_diag.shape)
                self.parallel_wrap_shot.comm.Allreduce(pseudo_h_diag, pseudo_h_diag_temp)
                pseudo_h_diag = pseudo_h_diag_temp

        # account for the measure in the integral over time
        r_norm2 *= self.solver.dt
        r_norm2 += self._regularization_fun(m0, dm_extend)*self.solver.dt * 2.0
        # The gradient is implemented as a time integral in TemporalModeling.adjoint_model(). I think the pseudo Hessian (F*F in notation Shin) also represents a time integral. So multiply with dt as well to be consistent.
        pseudo_h_diag *= self.solver.dt

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, 0.5*r_norm2)
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            aux_info['pseudo_hess_diag'] = (True, pseudo_h_diag)

        return grad

    def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError(
                "Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        result = m0.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:
                # Run the forward modeling step
                retval = self.modeling_tools.forward_model(shot, m0, return_parameters=['dWaveOp'])
                dWaveOp0 = retval['dWaveOp']

                linear_retval = self.modeling_tools.linear_forward_model(
                    shot, m0, m1, return_parameters=['simdata'], dWaveOp0=dWaveOp0)

                d1 = linear_retval['simdata']  # data from F applied to m1
                result += self.modeling_tools.migrate_shot(shot, m0, d1, dWaveOp=dWaveOp0)

        elif hessian_mode == 'full':
            for shot in shots:
                # Run the forward modeling step
                dWaveOp0 = list()  # wave operator derivative wrt model for u_0
                r0, adjoint_src = self._residual(shot, m0, dWaveOp=dWaveOp0, **kwargs)

                linear_retval = self.modeling_tools.linear_forward_model(
                    shot, m0, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
                d1 = linear_retval['simdata']
                dWaveOp1 = linear_retval['dWaveOp1']

                # <q, u1tt>, first adjointy bit
                dWaveOpAdj1 = []
                res1 = self.modeling_tools.migrate_shot(
                    shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
                result += res1

                # <p, u0tt>
                res2 = self.modeling_tools.migrate_shot(
                    shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
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
