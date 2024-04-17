import os

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PATH = os.path.dirname(os.path.abspath(__file__))

CSRCS = [
    os.path.join(PATH, "megalodon/csrc/blas.cc"),
    os.path.join(PATH, "megalodon/csrc/megalodon_extension.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/attention.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/attention_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/attention_softmax.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/attention_softmax_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/ema_hidden.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/ema_hidden_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/ema_parameters.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/ema_parameters_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/fftconv.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/fftconv_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/sequence_norm.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/sequence_norm_kernel.cu"),
    os.path.join(PATH, "megalodon/csrc/ops/timestep_norm.cc"),
    os.path.join(PATH, "megalodon/csrc/ops/timestep_norm_kernel.cu"),
]

INCLUDE_DIRS = [
    os.path.join(PATH, "megalodon/csrc"),
]

CXX_FLAGS = [
    "-O3",
    "-std=c++17",
]

NVCC_FLAGS = [
    # "-rdc=true",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # "--use_fast_math",
    "--threads",
    "4",
]


def main():
    setup(
        name='megalodon',
        version="0.0.1",
        ext_modules=[
            CUDAExtension("megalodon_extension",
                          CSRCS,
                          include_dirs=INCLUDE_DIRS,
                          extra_compile_args={
                              "cxx": CXX_FLAGS,
                              "nvcc": CXX_FLAGS + NVCC_FLAGS,
                          })
        ],
        cmdclass={'build_ext': BuildExtension},
    )


if __name__ == "__main__":
    main()
