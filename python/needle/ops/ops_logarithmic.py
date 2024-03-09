from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max_ori = array_api.max(Z, self.axes, keepdims=True)
        Z_max_red = array_api.max(Z, self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - Z_max_ori), axis=self.axes)) + Z_max_red
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_max = array_api.max(z.realize_cached_data(), self.axes, keepdims=True)
        z_exp = array_api.exp(z.realize_cached_data() - z_max)
        z_exp_sum = array_api.sum(z_exp, self.axes)
        if self.axes is None:
          return (out_grad / z_exp_sum).broadcast_to(z.shape) * z_exp
        expand_shape = list(z.shape)
        for axis in self.axes:
          expand_shape[axis] = 1
        return (out_grad / z_exp_sum).reshape(expand_shape).broadcast_to(z.shape) * z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

