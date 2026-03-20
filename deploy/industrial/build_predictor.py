"""Build the predictor megakernel for the 5-layer, 1024-dim code predictor.

Uses the pre-patched kernel.cu from csrc/ (vendored in this repo).
No external repo cloning needed.

Predictor dimensions (compile-time defaults in kernel.cu):
  - HIDDEN_SIZE: 1024
  - INTERMEDIATE_SIZE: 3072
  - NUM_Q_HEADS: 16, NUM_KV_HEADS: 8, HEAD_DIM: 128
"""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
_CSRC = os.path.join(_REPO_ROOT, "csrc")


def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None else default


def get_predictor_extension():
    """Build the predictor megakernel extension."""
    global _module
    if _module is not None:
        return _module

    kernel_cu = os.path.join(_CSRC, "kernel.cu")
    bindings_cpp = os.path.join(_CSRC, "torch_bindings.cpp")

    if not os.path.exists(kernel_cu):
        raise FileNotFoundError(
            f"kernel.cu not found at {kernel_cu}. "
            f"Ensure the repo csrc/ directory contains the vendored kernel source."
        )

    import torch
    cc = torch.cuda.get_device_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_blocks = min(sms, 128)

    KERNEL_FLAGS = [
        f"-DLDG_NUM_BLOCKS={num_blocks}",
        f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
        f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 16)}",
        f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
        f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
        f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
        "-DLDG_PREFETCH_QK=0",
        "-DLDG_PREFETCH_THREAD_STRIDE=10",
        "-DLDG_PREFETCH_DOWN=1",
        "-DLDG_PREFETCH_ELEM_STRIDE=1",
        "-DLDG_PREFETCH_BLOCK_STRIDE=1",
        "-DLDG_PREFETCH_GATE=1",
        "-DLDG_PREFETCH_UP=1",
        "-DLDG_VOCAB_SIZE=3072",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]

    CUDA_FLAGS = [
        "-O3", "--use_fast_math", "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-arch={arch}",
        f"-I{_CSRC}",
    ] + KERNEL_FLAGS

    print(f"Compiling predictor megakernel: HIDDEN=1024 INTER=3072 "
          f"blocks={num_blocks} arch={arch}")
    _module = load(
        name="qwen_megakernel_C",
        sources=[bindings_cpp, kernel_cu],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module
