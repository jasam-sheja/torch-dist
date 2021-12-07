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


class PdistFunction_optcompute(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        res = module.eff_pdist_forward(x)
        ctx.save_for_backward(x, res)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, res = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            grad = module.eff_pdist_backward(grad_output, x, res)

        return grad


class PdistFunction_optmem(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        res = module.eff_pdist_forward(x)
        ctx.save_for_backward(x)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            grad = module.eff_pdist_mem_backward(grad_output, x)

        return grad


class CdistSquareFunction(Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.device == x2.device
        res = module.eff_cdistsquare_forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x1, x2 = ctx.saved_tensors
        grad_x1 = grad_x2 = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x1, grad_x2 = module.eff_cdistsquare_backward(
                grad_output, x1, x2)
        elif ctx.needs_input_grad[0]:
            grad_x1 = module.eff_cdistsquare_x1_backward(
                grad_output, x1, x2)
        elif ctx.needs_input_grad[1]:
            grad_x2 = module.eff_cdistsquare_x2_backward(
                grad_output, x1, x2)

        return grad_x1, grad_x2


class PdistSquareFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        res = module.eff_pdistsquare_forward(x)
        ctx.save_for_backward(x)
        return res

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            grad = module.eff_pdistsquare_backward(grad_output, x)
        return grad


def cdist(x1: torch.Tensor, x2: torch.Tensor, *, opt: str = 'compute') -> torch.Tensor:
    r"""Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
        opt (str)  : focus optimization parameter. Either focus on `mem` for
                     for memory or `compute` for flops

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    Example:

        >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch_dist.euclidean.cdist(a, b)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """
    if opt == 'compute':
        return CdistFunction_optcompute.apply(x1, x2)
    elif opt == 'mem':
        return CdistFunction_optmem.apply(x1, x2)


def pdist(x: torch.Tensor, *, opt: str = 'compute') -> torch.Tensor:
    r"""Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x (Tensor): input tensor of shape :math:`B \times P \times M`.
        opt (str)  : focus optimization parameter. Either focus on `mem` for
                     for memory or `compute` for flops

    If x has shape :math:`B \times P \times M` then the
    output will have shape :math:`B \times P \times P`.

    Example:

        >>> a = torch.tensor([[[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
        >>> a
        tensor([[[ 0.9041,  0.0196],
                 [-0.3108, -2.4423],
                 [-0.4821,  1.0590]]])
        >>> torch_dist.euclidean.pdist(a)
        tensor([[[0.0000, 2.7453, 1.7326],
                 [2.7453, 0.0000, 3.5055],
                 [1.7326, 3.5055, 0.0000]]])
    """
    if opt == 'compute':
        return PdistFunction_optcompute.apply(x)
    elif opt == 'mem':
        return PdistFunction_optmem.apply(x)


def cdist_square(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    r"""Computes batched the squared 2-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (Tensor): input tensor of shape :math:`B \times R \times M`.

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    Example:

        >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch_dist.euclidean.cdistsquare(a, b)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """
    return CdistSquareFunction.apply(x1, x2)

def pdist_square(x: torch.Tensor) -> torch.Tensor:
    r"""Computes batched the squared 2-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (Tensor): input tensor of shape :math:`B \times R \times M`.

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    Example:

        >>> a = torch.tensor([[[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
        >>> a
        tensor([[[ 0.9041,  0.0196],
                 [-0.3108, -2.4423],
                 [-0.4821,  1.0590]]])
        >>> torch_dist.euclidean.pdist_square(a)
        tensor([[[ 0.0000,  7.5369,  3.0019],
                 [ 7.5369,  0.0000, 12.2884],
                 [ 3.0019, 12.2884,  0.0000]]])
    """
    return PdistSquareFunction.apply(x)
