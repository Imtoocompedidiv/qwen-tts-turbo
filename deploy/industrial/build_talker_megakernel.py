"""Build the megakernel for the talker (28-layer, 2048-dim) transformer.

The same kernel.cu code works for both predictor and talker — we just
change the model constants via preprocessor defines at compile time.

Key differences from predictor (code_predictor):
  - HIDDEN_SIZE: 1024 → 2048
  - INTERMEDIATE_SIZE: 3072 → 6144
  - NUM_KV_HEADS: 8 (same as original)
  - KV_SIZE: 1024 → 512

HEAD_DIM (128) and NUM_Q_HEADS (16) stay the same.
"""

import os, re, shutil
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v is not None else default


def _patch_kernel_for_talker(kernel_src):
    """Replace hardcoded constexpr values with preprocessor-overridable versions."""
    # Make model constants overridable via -D flags
    replacements = [
        ('constexpr int HIDDEN_SIZE = 1024;',
         '#ifndef MK_HIDDEN_SIZE\n#define MK_HIDDEN_SIZE 1024\n#endif\nconstexpr int HIDDEN_SIZE = MK_HIDDEN_SIZE;'),
        ('constexpr int INTERMEDIATE_SIZE = 3072;',
         '#ifndef MK_INTERMEDIATE_SIZE\n#define MK_INTERMEDIATE_SIZE 3072\n#endif\nconstexpr int INTERMEDIATE_SIZE = MK_INTERMEDIATE_SIZE;'),
        ('constexpr int NUM_Q_HEADS = 16;',
         '#ifndef MK_NUM_Q_HEADS\n#define MK_NUM_Q_HEADS 16\n#endif\nconstexpr int NUM_Q_HEADS = MK_NUM_Q_HEADS;'),
        ('constexpr int NUM_KV_HEADS = 8;',
         '#ifndef MK_NUM_KV_HEADS\n#define MK_NUM_KV_HEADS 8\n#endif\nconstexpr int NUM_KV_HEADS = MK_NUM_KV_HEADS;'),
    ]
    for old, new in replacements:
        kernel_src = kernel_src.replace(old, new)
    return kernel_src


def _add_nanosleep(kernel_src):
    """Add __nanosleep to all spin-wait loops (datacenter GPU fix)."""
    def _ns(m):
        indent = m.group(1)
        while_line = m.group(2).split('\n')[0]
        return f"{indent}{while_line}\n{indent}  __nanosleep(100);\n{indent}}}"
    return re.sub(r'^( +)(while \(.+?\) \{\n\s*\})', lambda m: _ns(m),
                  kernel_src, flags=re.MULTILINE)


def get_talker_extension(megakernel_path="/workspace/megakernel-tts"):
    """Build the talker megakernel extension. Separate from predictor."""
    global _module
    if _module is not None:
        return _module

    csrc = os.path.join(megakernel_path, "csrc")
    kernel_cu = os.path.join(csrc, "kernel.cu")

    # Create a patched copy for the talker
    talker_cu = os.path.join(csrc, "kernel_talker.cu")
    with open(kernel_cu) as f:
        src = f.read()
    src = _patch_kernel_for_talker(src)
    src = _add_nanosleep(src)
    with open(talker_cu, 'w') as f:
        f.write(src)

    # Auto-detect GPU arch
    import torch
    cc = torch.cuda.get_device_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_blocks = min(sms, 128)

    # Talker model constants
    TALKER_FLAGS = [
        "-DMK_HIDDEN_SIZE=2048",
        "-DMK_INTERMEDIATE_SIZE=6144",
        "-DMK_NUM_Q_HEADS=16",
        "-DMK_NUM_KV_HEADS=8",
        f"-DLDG_NUM_BLOCKS={num_blocks}",
        f"-DLDG_BLOCK_SIZE=512",
        # LM head not used for talker (we handle logits in Python)
        "-DLDG_LM_NUM_BLOCKS=16",
        "-DLDG_LM_BLOCK_SIZE=384",
        "-DLDG_LM_ROWS_PER_WARP=2",
        "-DLDG_ATTN_BLOCKS=16",  # NUM_Q_HEADS = 16
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

    print(f"Compiling talker megakernel: HIDDEN=2048 INTER=6144 KV_HEADS=8 blocks={num_blocks} arch={arch}")
    _module = load(
        name="qwen_megakernel_talker_C",
        sources=[
            os.path.join(csrc, "torch_bindings.cpp"),
            talker_cu,
        ],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{csrc}"],
        verbose=False,
    )
    return _module
