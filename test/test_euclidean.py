'''validate values against torch.cdist and gradients with gradcheck'''
import torch
import torch_dist
import pytest


class PTestCDist:
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


class PTestPDist:
    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("B", [1, 256],)
    @pytest.mark.parametrize("P", [1, 2, 3, 256],)
    @pytest.mark.parametrize("M", [1, 2, 3, 25, 256, 1000],)
    @pytest.mark.parametrize("scale", [0.1, 1, 10],)
    def test_inference(self, device, opt, B, P, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale

        gt = torch.cdist(a, a, compute_mode='use_mm_for_euclid_dist')
        d = torch_dist.euclidean.pdist(a, opt=opt)
        assert torch.allclose(
            gt, d, atol=1e-5), f"failed inference with diff {(gt-d).abs().max().item()}"

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("size", [(1, 2000, 5)],)
    def test_large_input(self, device, size, opt):
        x = torch.randn(size, device=device).float().requires_grad_()
        dist = torch_dist.euclidean.pdist(x, opt=opt)
        # Do a backward pass to check that it is valid for large matrices
        loss = dist.sum()
        loss.backward()

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("B", [1, 8],)
    @pytest.mark.parametrize("P", [1, 8],)
    @pytest.mark.parametrize("M", [1, 2, 32],)
    @pytest.mark.parametrize("scale", [0.01, 0.1, 1, 10, 100, 1000],)
    def test_gradrand(self, device, opt, B, P, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64) * scale
        a.requires_grad = True

        # # gradcheck is always failing
        # # main problem is when alpha in baddmm is -2
        # def func(a):
        #     return torch_dist.euclidean.pdist(a, opt=opt)
        # torch.autograd.gradcheck(func, a)

        # since pdist(a) is equivilant to cdist(a,a) use it to check gradients.
        d = torch_dist.euclidean.cdist(a, a)
        d.backward(torch.ones_like(d))
        ggt = a.grad

        a = a.detach().requires_grad_()
        d = torch_dist.euclidean.pdist(a, opt=opt)
        d.backward(torch.ones_like(d))
        gopt = a.grad
        assert ggt is not gopt
        assert torch.allclose(ggt, gopt)

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("opt", ['mem', 'compute'],)
    @pytest.mark.parametrize("B", [1, 8],)
    @pytest.mark.parametrize("P", [1, 8],)
    @pytest.mark.parametrize("M", [1, 2, 32],)
    @pytest.mark.parametrize("scale", [0.01, 0.1, 1, 10, 100, 1000],)
    def test_gradara(self, device, opt, B, P, M, scale):
        a = torch.arange(P * M).view(1, P, M).repeat(B, 1,
                                                     1).double().contiguous().to(device)
        a *= scale
        a.requires_grad = True

        # # gradcheck is always failing
        # # main problem is when alpha in baddmm is -2
        # def func(a):
        #     return torch_dist.euclidean.pdist(a, opt=opt)
        # torch.autograd.gradcheck(func, a)

        # since pdist(a) is equivilant to cdist(a,a) use it to check gradients.
        d = torch_dist.euclidean.cdist(a, a)
        d.backward(torch.ones_like(d))
        ggt = a.grad

        a = a.detach().requires_grad_()
        d = torch_dist.euclidean.pdist(a, opt=opt)
        d.backward(torch.ones_like(d))
        gopt = a.grad
        assert ggt is not gopt
        assert torch.allclose(ggt, gopt)


class PTestCDistSquare:
    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("B", [1, 256],)
    @pytest.mark.parametrize("P", [1, 2, 3, 256],)
    @pytest.mark.parametrize("R", [1, 2, 3, 256],)
    @pytest.mark.parametrize("M", [1, 2, 3, 25, 256, 1000],)
    @pytest.mark.parametrize("scale", [0.1, 1, 10],)
    def test_inference(self, device, B, P, R, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale
        b = torch.rand((B, R, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale

        gt = torch.cdist(a, b, compute_mode='use_mm_for_euclid_dist')**2
        d = torch_dist.euclidean.cdist_square(a, b)
        assert torch.allclose(
            gt, d, atol=1e-5), f"failed inference with diff {(gt-d).abs().max().item()}"

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("size", [(1, 2000, 5)],)
    def test_large_input(self, device, size):
        x = torch.randn(size, device=device, dtype=torch.float)
        y = torch.randn(size, device=device, dtype=torch.float)
        eps = 1e-6
        # to avoid extremum
        x = x - (((x - y) < eps).float() * 2 * eps)
        x.requires_grad = True
        y.requires_grad = True
        dist = torch_dist.euclidean.cdist_square(x, y)
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
        assert torch.autograd.gradcheck(
            torch_dist.euclidean.cdist_square, (a, b), raise_exception=False)


class TestPDistSquare:
    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("B", [1, 256],)
    @pytest.mark.parametrize("P", [1, 2, 3, 256],)
    @pytest.mark.parametrize("M", [1, 2, 3, 25, 256, 1000],)
    @pytest.mark.parametrize("scale", [0.1, 1, 10],)
    def test_inference(self, device, B, P, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64, requires_grad=False) * scale

        gt = torch.cdist(a, a, compute_mode='use_mm_for_euclid_dist')**2
        d = torch_dist.euclidean.pdist_square(a)
        assert torch.allclose(
            gt, d, atol=1e-5), f"failed inference with diff {(gt-d).abs().max().item()}"

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("size", [(1, 2000, 5)],)
    def test_large_input(self, device, size):
        x = torch.randn(size, device=device).float().requires_grad_()
        dist = torch_dist.euclidean.pdist_square(x)
        # Do a backward pass to check that it is valid for large matrices
        loss = dist.sum()
        loss.backward()

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("B", [1, 8],)
    @pytest.mark.parametrize("P", [1, 8],)
    @pytest.mark.parametrize("M", [1, 2, 32],)
    @pytest.mark.parametrize("scale", [0.01, 0.1, 1, 10, 100, 1000],)
    def test_gradrand(self, device, B, P, M, scale):
        a = torch.rand((B, P, M), device=device,
                       dtype=torch.float64) * scale
        a.requires_grad = True

        # since pdist(a) is equivilant to cdist(a,a) use it to check gradients.
        d = torch_dist.euclidean.cdist_square(a, a)
        d.backward(torch.ones_like(d))
        ggt = a.grad

        a = a.detach().requires_grad_()
        d = torch_dist.euclidean.pdist_square(a)
        d.backward(torch.ones_like(d))
        gopt = a.grad
        assert ggt is not gopt
        assert torch.allclose(ggt, gopt)

    @pytest.mark.parametrize(
        "device", ['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    )
    @pytest.mark.parametrize("B", [1, 8],)
    @pytest.mark.parametrize("P", [1, 8],)
    @pytest.mark.parametrize("M", [1, 2, 32],)
    @pytest.mark.parametrize("scale", [0.01, 0.1, 1, 10, 100, 1000],)
    def test_gradara(self, device, B, P, M, scale):
        a = torch.arange(P * M).view(1, P, M).repeat(B, 1,
                                                     1).double().contiguous().to(device)
        a *= scale
        a.requires_grad = True

        # since pdist(a) is equivilant to cdist(a,a) use it to check gradients.
        d = torch_dist.euclidean.cdist_square(a, a)
        d.backward(torch.ones_like(d))
        ggt = a.grad

        a = a.detach().requires_grad_()
        d = torch_dist.euclidean.pdist_square(a)
        d.backward(torch.ones_like(d))
        gopt = a.grad
        assert ggt is not gopt
        assert torch.allclose(ggt, gopt)
