import time

import torch


def _get_gpu_mem(synchronize=True, empty_cache=False, reset_peak=False):
    if synchronize:
        torch.cuda.synchronize()
    if empty_cache:
        torch.cuda.empty_cache()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    mem_all = torch.cuda.memory_allocated()
    mem_cached = torch.cuda.memory_reserved()
    mem_peak = torch.cuda.max_memory_allocated()
    return mem_all, mem_cached, mem_peak


class mem_context:
    def __init__(self, synchronize=True, empty_cache=False, reset_peak=False) -> None:
        self.synchronize = synchronize
        self.empty_cache = empty_cache
        self.reset_peak = reset_peak

        self.log0 = dict(mem_all=None, mem_cached=None, mem_peak=None)
        self.log1 = dict(mem_all=None, mem_cached=None, mem_peak=None)

    def __enter__(self):
        mem_all, mem_cached, mem_peak = _get_gpu_mem(
            self.synchronize, self.empty_cache, self.reset_peak)
        self.log0 = dict(
            mem_all=mem_all, mem_cached=mem_cached, mem_peak=mem_peak)
        self._start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *args, **kw):
        mem_all, mem_cached, mem_peak = _get_gpu_mem(self.synchronize)
        self._end_time = time.perf_counter_ns()
        self.log1 = dict(
            mem_all=mem_all, mem_cached=mem_cached, mem_peak=mem_peak)

    @property
    def mem_peak(self):
        return (self.log1['mem_peak'] - self.log0['mem_peak'])

    @property
    def time(self):
        return self._end_time - self._start_time

    def __str__(self):
        mem_all = (self.log1['mem_all'] - self.log0['mem_all']) // 2**20
        mem_cached = self.log0['mem_cached'] // 2**20, self.log1['mem_cached'] // 2**20
        mem_peak = (self.log1['mem_peak'] - self.log0['mem_peak']) // 2**20
        return f"mem_all={mem_all}MB, mem_cached={mem_cached}MB, mem_peak={mem_peak}MB"
