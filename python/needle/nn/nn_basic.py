"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        if bias:
          self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, requires_grad=True), (1,out_features)))
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X.matmul(self.weight) 
        if self.bias:
          y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_exp_sum = ops.logsumexp(logits, axes=(1,)).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (log_exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          E_x_run = ops.summation(x, axes=(0,)) / x.shape[0]
          E_x = ops.reshape(ops.summation(x, axes=(0,)), (1, x.shape[1])) / x.shape[0]
          Var_x_run = ops.summation((x - ops.broadcast_to(E_x, x.shape))**2, axes=(0,)) / x.shape[0]
          Var_x = ops.reshape(ops.summation((x - ops.broadcast_to(E_x, x.shape))**2, axes=(0,)), (1, x.shape[1])) / x.shape[0]
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E_x_run.data
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_x_run.data
          nume = x - ops.broadcast_to(E_x, x.shape)
          deno = (ops.broadcast_to(Var_x, x.shape) + self.eps)**0.5
        else:
          nume = x - ops.broadcast_to(self.running_mean, x.shape)
          deno = (ops.broadcast_to(self.running_var, x.shape) + self.eps)**0.5
        return ops.broadcast_to(self.weight, x.shape) * nume / deno + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E_x = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(1,)), (x.shape[0], 1)) / x.shape[1], x.shape)
        Var_x = ops.broadcast_to(ops.reshape(ops.summation((x - E_x)**2, axes=(1,)), (x.shape[0], 1)) / x.shape[1], x.shape)
        return ops.broadcast_to(self.weight, x.shape) * (x - E_x) / (Var_x + self.eps)**0.5 + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mask = init.randb(*x.shape, p=1-self.p)
          return mask * x / (1 - self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
