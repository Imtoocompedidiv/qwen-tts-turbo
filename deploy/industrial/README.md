# Industrial Version — Sub-millisecond TTFP

This is the performance-critical version with C++/CUDA optimizations.

## Optimizations over the Python version

1. **Fused mega-graph**: sampling + predictor (15 steps) + first talker decode captured as a single CUDA graph replay
2. **Triton sampling kernel**: fused top-k + temperature + sampling in one kernel launch
3. **Zero-copy codec output**: pre-allocated output buffer, no torch.cat per step
4. **C++ hot path**: request parsing → generation loop in C++ via pybind11, eliminates Python interpreter overhead

## Target: <1ms server TTFP (vs 2.6ms Python version)
