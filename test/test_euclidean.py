'''validate values against torch.cdist and gradients with gradcheck'''
import torch
import torch_dist
import pytest


class TestCDist:
    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("B", [1, 256],)
    @pytest.mark.parametrize("P", [1, 2, 3, 256],)
    @pytest.mark.parametrize("R", [1, 2, 3, 256],)
    @pytest.mark.parametrize("M", [1, 2, 3, 25, 256, 1000],)
    @pytest.mark.parametrize("scale", [0.1, 1, 10],)
    def test_inference(self, device, opt, B, P, R, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale
        b = torch.rand((B, R, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale

        gt = torch.cdist(a, b, compute_mode='use_mm_for_euclid_dist')
        d = torch_dist.euclidean.cdist(a, b, opt=opt)
        assert torch.allclose(
            gt, d, atol=1e-5), f"failed inference with diff {(gt-d).abs().max().item()}"

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("size", [(1, 2000, 5)],)
    def test_large_input(self, device, size, opt):
        x = torch.randn(size, device=device, dtype=torch.float)
        y = torch.randn(size, device=device, dtype=torch.float)
        eps = 1e-6
        # to avoid extremum
        x = x - (((x - y) < eps).float() * 2 * eps)
        x.requires_grad = True
        y.requires_grad = True
        dist = torch_dist.euclidean.cdist(x, y, opt=opt)
        # Do a backward pass to check that it is valid for large
        # matrices
        loss = dist.sum()
        loss.backward()

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("B", [1, 8],)
    @pytest.mark.parametrize("P", [1, 8],)
    @pytest.mark.parametrize("R", [1, 8],)
    @pytest.mark.parametrize("M", [1, 2, 32],)
    @pytest.mark.parametrize("scale", [0.1, 1, 10],)
    def test_grad(self, device, B, P, R, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64, requires_grad=True) * scale
        b = torch.rand((B, R, M), device=device,
                       dtype=torch.float64, requires_grad=True) * scale

        def optmem(a, b):
            return torch_dist.euclidean.cdist(a, b, opt='mem')

        def optcompute(a, b):
            return torch_dist.euclidean.cdist(a, b, opt='compute')
        assert torch.autograd.gradcheck(optmem, (a, b), raise_exception=False)
        assert torch.autograd.gradcheck(
            optcompute, (a, b), raise_exception=False)
