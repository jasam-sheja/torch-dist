import colorsys
from collections import defaultdict, namedtuple
from typing import Callable

import numpy as np
import torch
import torch.jit
from matplotlib import colors
from matplotlib import pyplot as plt
from mem import mem_context
from torch.utils.benchmark import Timer
from tqdm import tqdm

import torch_dist


def _scale_lightness(rgb, scale_l):
    '''https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib/49601444'''
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def run(func, *input, ret_mem=False, verbose=False):
    name = getattr(func, '__name__', None) or type(func)

    t = Timer(
        stmt='func(*input)',
        globals={'func': func, 'input': input}
    )
    m = t.blocked_autorange(min_run_time=1)
    if verbose:
        print(m)
    if not ret_mem:
        return m.median,

    with mem_context(empty_cache=True, reset_peak=True) as cxt:
        out = func(*input)
    fwd_peak = cxt.mem_peak
    if verbose == 'cuda' or verbose == True:
        print(f"""{name}:fwd: peak {str(cxt)}""")
    with mem_context(empty_cache=True, reset_peak=True) as cxt:
        out.backward(torch.ones_like(out))
    bwd_peak = cxt.mem_peak
    if verbose == 'cuda' or verbose == True:
        print(f"""{name}:bwd: peak {str(cxt)}""")

    with mem_context(empty_cache=True, reset_peak=True) as cxt:
        out = func(*input)
        out.backward(torch.ones_like(out))
    if verbose == 'cuda' or verbose == True:
        print(f"""{name}:fwd+bwd: peak {str(cxt)}""")
    peak = cxt.mem_peak
    return m.median, fwd_peak, bwd_peak, peak


def compare_cdist(func1: Callable, func2: Callable, device: str, *, verbose=False, num=1000, func1_name: str = None, func2_name: str = None):
    import torch
    import torch.jit
    from torch.nn import functional as F

    # from gait_tools.eval.distance import cdist2
    from torch.utils.benchmark import Timer

    if func1_name is None:
        func1_name = getattr(func1, '__name__', None) or func1.__class__
    if func2_name is None:
        func2_name = getattr(func2, '__name__', None) or func2.__class__

    history = defaultdict(lambda: {
        func1_name: [],
        func2_name: [],
    })

    a = torch.rand(1000, 1000).requires_grad_().to(device)
    b = torch.rand(1000, 1000).requires_grad_().to(device)

    experiments = [f'({a.shape[0]}x{a.shape[1]}), (Xx{b.shape[1]})',
                   f'(Xx{a.shape[1]}), ({b.shape[0]}x{b.shape[1]})',
                   f'(Xx{a.shape[1]}), (Xx{b.shape[1]})',
                   f'({a.shape[0]}xX), ({b.shape[0]}xX)',
                   ]
    N = np.unique(np.geomspace(26, 1000, num=num,
                  endpoint=True).astype(np.int32))
    abslices = namedtuple('abslices', ['ai', 'aj', 'bi', 'bj'])
    for n in tqdm(N):
        slices = {
            experiments[0]: abslices(slice(0, None), slice(0, None),
                                     slice(0, n), slice(0, None)),
            experiments[1]: abslices(slice(0, n), slice(0, None),
                                     slice(0, None), slice(0, None)),
            experiments[2]: abslices(slice(0, n), slice(0, None),
                                     slice(0, n), slice(0, None)),
            experiments[3]: abslices(slice(0, None), slice(0, n),
                                     slice(0, None), slice(0, n)),
        }
        for exp, sli in slices.items():
            ai = a[sli.ai, sli.aj].unsqueeze(0).contiguous()
            bi = b[sli.bi, sli.bj].unsqueeze(0).contiguous()
            history[exp][func1_name].append(
                run(func1, ai, bi, ret_mem=(device != 'cpu'), verbose=verbose))
            history[exp][func2_name].append(
                run(func2, ai, bi, ret_mem=(device != 'cpu'), verbose=verbose))

    # SpeedUP
    for expname, data in history.items():
        fig, ax = plt.subplots(figsize=(19.20, 12.80))
        t0, *_ = zip(*data[func1_name])
        t1, *_ = zip(*data[func2_name])
        color = next(ax._get_lines.prop_cycler)['color']
        color = colors.ColorConverter.to_rgb(color)
        plt.plot(N, np.divide(t0, t1), color=color)
        plt.scatter(N, np.divide(t0, t1), color=color, label=device)
        plt.title(expname)
        plt.xlabel('X')
        plt.ylabel('Speedup')
        plt.legend()
        plt.grid()
        plt.savefig(f'{func1_name}-{func2_name}-{expname}-speedup.png')
        plt.close()
    if device == 'cpu':
        return
    # Mem
    for expname, data in history.items():
        fig, ax = plt.subplots(figsize=(19.20, 12.80))
        _, fwd0, bwd0, peak0 = zip(*data[func1_name])
        _, fwd1, bwd1, peak1 = zip(*data[func2_name])
        color = next(ax._get_lines.prop_cycler)['color']
        color = colors.ColorConverter.to_rgb(color)

        plt.plot(N, np.divide(np.subtract(peak0, peak1), peak0), color=color)
        plt.scatter(N, np.divide(np.subtract(peak0, peak1), peak0),
                    color=color, label='both')
        fwdcolor = _scale_lightness(color, 1.5)
        plt.plot(N, np.divide(np.subtract(fwd0, fwd1), fwd0), color=fwdcolor)
        plt.scatter(N, np.divide(np.subtract(fwd0, fwd1), fwd0),
                    color=fwdcolor, label='forward')
        bwdcolor = _scale_lightness(color, 0.5)
        plt.plot(N, np.divide(np.subtract(bwd0, bwd1), bwd0), color=bwdcolor)
        plt.scatter(N, np.divide(np.subtract(bwd0, bwd1), bwd0),
                    color=bwdcolor, label='backward')
        plt.title(expname)
        plt.xlabel('X')
        plt.ylabel('Mem saved ratio')
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(f'{func1_name}-{func2_name}-{expname}-memory.png')
        plt.close()


if __name__ == '__main__':
    compare_cdist(torch.cdist, torch_dist.euclidean.CdistFunction_optmem.apply, 'cuda',
                  num=10, func1_name='torch.dist',
                  func2_name='torch_dist.euclidean.cdist_mem')
    compare_cdist(torch.cdist, torch_dist.euclidean.CdistFunction_optcompute.apply, 'cuda',
                  num=10, func1_name='torch.dist',
                  func2_name='torch_dist.euclidean.cdist_compute')
