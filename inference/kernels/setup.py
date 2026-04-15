"""Build script for fused CUDA kernels (TurboQuant + GPTQ)."""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

_nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_86,code=sm_86",   # 3090
    "-gencode=arch=compute_89,code=sm_89",   # 4090
    "-gencode=arch=compute_90,code=sm_90",   # H100/H20
    "-lineinfo",
]

setup(
    name="glorcq_kernels",
    ext_modules=[
        CUDAExtension(
            name="_turbo_matmul_cuda",
            sources=[
                os.path.join(this_dir, "turbo_matmul.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": _nvcc_flags,
            },
        ),
        CUDAExtension(
            name="_gptq_matmul_cuda",
            sources=[
                os.path.join(this_dir, "gptq_matmul.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": _nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
