

from pysit.util.parallel import ParallelWrapShotNull

__all__ = ['ObjectiveFunctionBase']

__docformat__ = "restructuredtext en"

class ObjectiveFunctionBase(object):

    def __init__(self, solver, WaveCompressInfo=None, parallel_wrap_shot=ParallelWrapShotNull()):
        self.solver = solver
        self.modeling_tools = None
        self.WaveCompressInfo = WaveCompressInfo

        self.parallel_wrap_shot = parallel_wrap_shot

    def use_parallel(self):
        return self.parallel_wrap_shot.use_parallel # or self.parallem_wrap_solver.useparallel etc

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def compute_gradient(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def apply_hessian(self, operand, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")