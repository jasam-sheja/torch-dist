import torch
from torch.autograd import Function

import torch_dist._C.euclidean as module

__all__ = [
    'cdist'
]
class CdistFunction_optcompute(Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.device == x2.device
        if x1.is_cuda:
            res = module.eff2_cdist_forward(x1, x2)
        else:
            res = module.eff_cdist_forward(x1, x2)
        ctx.save_for_backward(x1, x2, res)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x1, x2, res = ctx.saved_tensors
        grad_x1 = grad_x2 = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x1, grad_x2 = module.eff_cdist_backward(
                grad_output, x1, x2, res)
        elif ctx.needs_input_grad[0]:
            grad_x1 = module.eff_cdist_x1_backward(
                grad_output, x1, x2, res)
        elif ctx.needs_input_grad[1]:
            grad_x2 = module.eff_cdist_x2_backward(
                grad_output, x1, x2, res)

        return grad_x1, grad_x2


class CdistFunction_optmem(Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.device == x2.device
        if x1.is_cuda:
            res = module.eff2_cdist_forward(x1, x2)
        else:
            res = module.eff_cdist_forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x1, x2 = ctx.saved_tensors
        grad_x1 = grad_x2 = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x1, grad_x2 = module.eff_cdist_mem_backward(
                grad_output, x1, x2)
        elif ctx.needs_input_grad[0]:
            grad_x1 = module.eff_cdist_x1_mem_backward(
                grad_output, x1, x2)
        elif ctx.needs_input_grad[1]:
            grad_x2 = module.eff_cdist_x2_mem_backward(
                grad_output, x1, x2)

        return grad_x1, grad_x2


def cdist(x1: torch.Tensor, x2: torch.Tensor, opt: str = 'compute') -> torch.Tensor:
    if opt == 'compute':
        return CdistFunction_optcompute.apply(x1, x2)
    elif opt == 'mem':
        return CdistFunction_optmem.apply(x1, x2)
