"""Build the talker megakernel for the 28-layer, 2048-dim transformer.

Uses the pre-patched kernel_talker.cu from csrc/ (vendored in this repo).
No external repo cloning needed — the CUDA source is self-contained.

Compile-time overrides for talker dimensions:
  - HIDDEN_SIZE: 1024 → 2048
  - INTERMEDIATE_SIZE: 3072 → 6144
  - NUM_KV_HEADS: 8 (same as original)

HEAD_DIM (128) and NUM_Q_HEADS (16) stay the same.
"""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
_CSRC = os.path.join(_REPO_ROOT, "csrc")


def get_talker_extension(megakernel_path=None):
    """Build the talker megakernel extension.

    Args:
        megakernel_path: Ignored (kept for backward compat). Uses csrc/ from repo root.
    """
    global _module
    if _module is not None:
        return _module

    csrc = _CSRC
    talker_cu = os.path.join(csrc, "kernel_talker.cu")
    bindings_cpp = os.path.join(csrc, "torch_bindings.cpp")

    if not os.path.exists(talker_cu):
        raise FileNotFoundError(
            f"kernel_talker.cu not found at {talker_cu}. "
            f"Ensure the repo csrc/ directory contains the vendored kernel source."
        )

    import torch
    cc = torch.cuda.get_device_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_blocks = min(sms, 128)

    TALKER_FLAGS = [
        "-DMK_HIDDEN_SIZE=2048",
        "-DMK_INTERMEDIATE_SIZE=6144",
        "-DMK_NUM_Q_HEADS=16",
        "-DMK_NUM_KV_HEADS=8",
        f"-DLDG_NUM_BLOCKS={num_blocks}",
        "-DLDG_BLOCK_SIZE=512",
        "-DLDG_LM_NUM_BLOCKS=16",
        "-DLDG_LM_BLOCK_SIZE=384",
        "-DLDG_LM_ROWS_PER_WARP=2",
        "-DLDG_ATTN_BLOCKS=16",
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
        f"-I{csrc}",
    ] + TALKER_FLAGS

    print(f"Compiling talker megakernel: HIDDEN=2048 INTER=6144 KV_HEADS=8 "
          f"blocks={num_blocks} arch={arch}")
    _module = load(
        name="qwen_megakernel_talker_C",
        sources=[bindings_cpp, talker_cu],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{csrc}"],
        verbose=False,
    )
    return _module
