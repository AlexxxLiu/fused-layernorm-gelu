from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ln_gelu",
    ext_modules=[
        CUDAExtension(
            name="fused_ln_gelu",
            sources=["fused_layernorm_gelu.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                # -lineinfo keeps source mapping so ncu can attribute stalls to
                # actual lines. Costs nothing at runtime.
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
