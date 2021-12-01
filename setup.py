import sys
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extensions():
    this_dir = Path(__file__).parent
    extension_dir = this_dir / 'torch_dist' / 'csrc'
    source = list(extension_dir.glob('*.cpp'))
    source += list(extension_dir.glob('*.h'))
    source += list(extension_dir.glob('*.hpp'))
    print(source, file=sys.stderr)
    source = set(map(str, source))
    return [CppExtension(name='torch_dist._C', sources=sorted(source))]


setup(
    name='torch_dist',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    }
)
