"""
Patch megakernel for datacenter GPUs (A100/H100/B200).

Only change: add __nanosleep(100) in all empty spin-wait loops to prevent
scheduler starvation on datacenter GPUs where blocks > SMs can cause
the GPU scheduler to suspend spinning blocks indefinitely.

fence.acq_rel.gpu PTX is KEPT — it's supported on sm_70+ (A100=sm_80,
H100=sm_90) and provides acquire-release semantics that __threadfence
does NOT (threadfence is release-only, causing stale reads and deadlock).

NO cooperative_groups, NO -rdc — keeps <<<>>> launch syntax.

Usage: python patch_kernel_datacenter.py /path/to/megakernel-tts
"""
import os, sys, re


def patch_kernel(kernel_path):
    with open(kernel_path) as f:
        src = f.read()

    # 1. Add adaptive spin-wait in ALL empty spin-wait loops to prevent deadlock
    # On datacenter GPUs, the scheduler may suspend spinning blocks, causing deadlock.
    # Strategy: tight spin first (fast convergence), nanosleep only as fallback.
    # Most barriers converge in <100 iterations, so nanosleep rarely fires.
    def add_nanosleep(m):
        indent = m.group(1)
        while_line = m.group(2).split('\n')[0]  # "while (...) {"
        return (f"{indent}{{ int _spin = 0;\n"
                f"{indent}{while_line}\n"
                f"{indent}  if (++_spin > 64) __nanosleep(32);\n"
                f"{indent}}}}}")

    src = re.sub(
        r'^( +)(while \(.+?\) \{\n\s*\})',
        lambda m: add_nanosleep(m),
        src,
        flags=re.MULTILINE
    )
    n2 = src.count('__nanosleep')
    print(f"  __nanosleep: inserted in {n2} spin-wait loops")

    # 2. Verify fence.acq_rel is preserved (critical for correctness)
    n_fence = src.count('fence.acq_rel.gpu')
    print(f"  fence.acq_rel.gpu: {n_fence} occurrences preserved")

    with open(kernel_path, 'w') as f:
        f.write(src)
    print(f"Patched {kernel_path}: __nanosleep in {n2} spin-waits")


def patch_build(build_path):
    with open(build_path) as f:
        bsrc = f.read()

    # Auto-detect arch
    try:
        import torch
        cc = torch.cuda.get_device_capability()
        arch = f"sm_{cc[0]}{cc[1]}"
        sms = torch.cuda.get_device_properties(0).multi_processor_count
    except:
        arch, sms = "sm_80", 108

    bsrc = bsrc.replace('"-arch=sm_120a",', f'"-arch={arch}",')
    # Use up to 128 blocks — all fit on H100 (132 SMs), more parallelism = faster matmuls
    num_blocks = min(sms, 128)
    bsrc = bsrc.replace(
        "_env_int('LDG_NUM_BLOCKS', 128)",
        f"_env_int('LDG_NUM_BLOCKS', {num_blocks})"
    )

    # Keep LDG_WEIGHT_LDCS — ld.no_allocate works on sm_70+ (H100 = sm_90)

    with open(build_path, 'w') as f:
        f.write(bsrc)
    print(f"Patched {build_path}: arch={arch}, blocks={num_blocks}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/workspace/megakernel-tts"
    patch_kernel(os.path.join(base, "csrc", "kernel.cu"))
    patch_build(os.path.join(base, "qwen_megakernel", "build_tts.py"))
