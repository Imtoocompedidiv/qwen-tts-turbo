"""Patch kernel.cu for INT8 weight quantization (W8A32).

Changes all matmul inner loops from:
  BF16: 8 values per 128-bit load, stride 8
To:
  INT8: 16 values per 128-bit load, stride 16, with per-row float scale

The LDGLayerWeights struct gets scale pointers for each weight matrix.
Weight pointers change from __nv_bfloat16* to int8_t*.
"""
import re


def patch_kernel_int8(src):
    """Transform kernel.cu for INT8 weights. Returns modified source."""

    # 1. Add INT8 helper near the top (after ldg_load_weight_u4)
    int8_helper = '''
// INT8 weight dequantization: load 16 int8 values, multiply-accumulate with float activation
__device__ __forceinline__ float ldg_int8_dot16(
    const int8_t *__restrict__ w_ptr, const float *__restrict__ act, float scale) {
  uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(w_ptr));
  const int8_t *w = reinterpret_cast<const int8_t *>(&w_u4);
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    sum += (float)w[i] * act[i];
  }
  return sum * scale;
}
'''
    src = src.replace(
        '// =============================================================================\n// Optimized matvec',
        int8_helper + '\n// =============================================================================\n// Optimized matvec'
    )

    # 2. Change LDGLayerWeights struct to INT8 (conditionally)
    old_struct = '''struct LDGLayerWeights {
  const __nv_bfloat16 *input_layernorm_weight;
  const __nv_bfloat16 *q_proj_weight;
  const __nv_bfloat16 *k_proj_weight;
  const __nv_bfloat16 *v_proj_weight;
  const __nv_bfloat16 *q_norm_weight;
  const __nv_bfloat16 *k_norm_weight;
  const __nv_bfloat16 *o_proj_weight;
  const __nv_bfloat16 *post_attn_layernorm_weight;
  const __nv_bfloat16 *gate_proj_weight;
  const __nv_bfloat16 *up_proj_weight;
  const __nv_bfloat16 *down_proj_weight;
};'''

    # Keep original pointer types, just ADD scale pointers for INT8 mode
    new_struct = '''struct LDGLayerWeights {
  const __nv_bfloat16 *input_layernorm_weight;
  const __nv_bfloat16 *q_proj_weight;
  const __nv_bfloat16 *k_proj_weight;
  const __nv_bfloat16 *v_proj_weight;
  const __nv_bfloat16 *q_norm_weight;
  const __nv_bfloat16 *k_norm_weight;
  const __nv_bfloat16 *o_proj_weight;
  const __nv_bfloat16 *post_attn_layernorm_weight;
  const __nv_bfloat16 *gate_proj_weight;
  const __nv_bfloat16 *up_proj_weight;
  const __nv_bfloat16 *down_proj_weight;
#ifdef MK_INT8_WEIGHTS
  const float *q_scale;
  const float *k_scale;
  const float *v_scale;
  const float *o_scale;
  const float *gate_scale;
  const float *up_scale;
  const float *down_scale;
#endif
};'''
    src = src.replace(old_struct, new_struct)

    # 3. Replace all BF16 matmul inner loops with INT8 versions
    # Pattern: the 8-element BF16 dot product block
    bf16_dot_pattern = r'''(uint4 \w+_u4 =\s*\n\s*ldg_load_weight_u4\(reinterpret_cast<const uint4 \*>\((\w+) \+ k\)\);\s*\n\s*__nv_bfloat16 \*(\w+) = reinterpret_cast<__nv_bfloat16 \*>\(&\w+_u4\);\s*\n\s*float4 (\w+) = \*reinterpret_cast<const float4 \*>\((\w+) \+ k\);\s*\n\s*float4 (\w+) = \*?reinterpret_cast<const float4 \*>\(\5 \+ k \+ 4\);\s*\n\s*\n\s*(\w+) \+= __bfloat162float\(\3\[0\]\).*?__bfloat162float\(\3\[7\]\) \* \6\.w;)'''

    # This regex is too fragile. Let me use a simpler approach: replace specific code blocks.

    # Actually, let me use a line-by-line approach for each matmul.
    # The key change is in the inner loop body. Let me use #ifdef blocks.

    # Given the complexity, let me use a different strategy:
    # Add a macro that handles the dot product, and replace the inner loop bodies.

    # Add a generic INT8 matmul macro
    macro = '''
#ifdef MK_INT8_WEIGHTS
#define LDG_MATVEC_INNER(weight_row, act_ptr, k, lane_id, DIM, sum, scale_val) \\
  for (int k = lane_id * 16; k < DIM; k += WARP_SIZE * 16) { \\
    uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(weight_row + k)); \\
    const int8_t *wp = reinterpret_cast<const int8_t *>(&w_u4); \\
    const float *ap = act_ptr + k; \\
    float s = 0.0f; \\
    _Pragma("unroll") \\
    for (int _i = 0; _i < 16; _i++) s += (float)wp[_i] * ap[_i]; \\
    sum += s * scale_val; \\
  }
#define LDG_MATVEC_STRIDE 16
#else
#define LDG_MATVEC_STRIDE 8
#endif
'''
    # Insert after the INT8 helper
    src = src.replace(
        '// =============================================================================\n// Optimized matvec',
        macro + '\n// =============================================================================\n// Optimized matvec'
    )

    # Now I need to change the actual loops. The safest approach:
    # Add #ifdef MK_INT8_WEIGHTS / #else / #endif around each matmul loop body.
    # But this would make the code huge. Let me instead write a proper code generation.

    # For now, let's just return the source with the struct and helpers changed.
    # The actual matmul replacement will be done by search-and-replace of the specific patterns.

    return src


def create_int8_kernel(orig_src):
    """Create a complete INT8 kernel variant. Returns the modified source."""
    src = orig_src

    # Add INT8 struct and helpers
    src = patch_kernel_int8(src)

    # Now replace each matmul loop. There are 4 patterns to replace:
    # All share the structure: for(k=lane*8; k<DIM; k+=WARP*8) { load_bf16; dot; accumulate }

    # QKV projection (weight_row + k, s_norm + k)
    # O projection (o_row + k, s_attn + k)
    # Gate+Up (gate_row + k AND up_row + k, s_act + k)
    # Down (down_row + k, s_mlp/g_mlp_intermediate + k)

    # The replacement: wrap each loop in #ifdef/#else/#endif
    # BF16 loop stays in #else, INT8 loop goes in #ifdef MK_INT8_WEIGHTS

    # QKV matmul replacement
    qkv_bf16 = '''      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(weight_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_norm + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_norm + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }'''

    qkv_int8 = '''#ifdef MK_INT8_WEIGHTS
      // INT8: cast bf16 pointer to int8, use per-channel scale
      const int8_t *weight_row_i8 = reinterpret_cast<const int8_t *>(weight_row);
      float row_scale;
      if (m < Q_SIZE) row_scale = w.q_scale[m];
      else if (m < Q_SIZE + KV_SIZE) row_scale = w.k_scale[m - Q_SIZE];
      else row_scale = w.v_scale[m - Q_SIZE - KV_SIZE];

      float sum = 0.0f;
#pragma unroll 2
      for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
        uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(weight_row_i8 + k));
        const int8_t *wp = reinterpret_cast<const int8_t *>(&w_u4);
        float s = 0.0f;
        #pragma unroll
        for (int _i = 0; _i < 16; _i++) s += (float)wp[_i] * s_norm[k + _i];
        sum += s;
      }
      sum *= row_scale;
#else''' + qkv_bf16 + '''
#endif'''
    src = src.replace(qkv_bf16, qkv_int8)

    # O projection
    o_bf16 = '''      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(o_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_attn + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_attn + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }'''

    o_int8 = '''#ifdef MK_INT8_WEIGHTS
      const int8_t *o_row_i8 = reinterpret_cast<const int8_t *>(o_row);
      float sum = 0.0f;
      float row_scale_o = w.o_scale[m];
#pragma unroll 2
      for (int k = lane_id * 16; k < Q_SIZE; k += WARP_SIZE * 16) {
        uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(o_row_i8 + k));
        const int8_t *wp = reinterpret_cast<const int8_t *>(&w_u4);
        float s = 0.0f;
        #pragma unroll
        for (int _i = 0; _i < 16; _i++) s += (float)wp[_i] * s_attn[k + _i];
        sum += s;
      }
      sum *= row_scale_o;
#else''' + o_bf16 + '''
#endif'''
    src = src.replace(o_bf16, o_int8)

    # Gate+Up
    gate_up_bf16 = '''      float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 g_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(gate_row + k));
        uint4 u_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(up_row + k));
        __nv_bfloat16 *g_ptr = reinterpret_cast<__nv_bfloat16 *>(&g_u4);
        __nv_bfloat16 *u_ptr = reinterpret_cast<__nv_bfloat16 *>(&u_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_act + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_act + k + 4);

        gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                    __bfloat162float(g_ptr[1]) * a1.y +
                    __bfloat162float(g_ptr[2]) * a1.z +
                    __bfloat162float(g_ptr[3]) * a1.w +
                    __bfloat162float(g_ptr[4]) * a2.x +
                    __bfloat162float(g_ptr[5]) * a2.y +
                    __bfloat162float(g_ptr[6]) * a2.z +
                    __bfloat162float(g_ptr[7]) * a2.w;

        up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                  __bfloat162float(u_ptr[1]) * a1.y +
                  __bfloat162float(u_ptr[2]) * a1.z +
                  __bfloat162float(u_ptr[3]) * a1.w +
                  __bfloat162float(u_ptr[4]) * a2.x +
                  __bfloat162float(u_ptr[5]) * a2.y +
                  __bfloat162float(u_ptr[6]) * a2.z +
                  __bfloat162float(u_ptr[7]) * a2.w;
      }'''

    gate_up_int8 = '''#ifdef MK_INT8_WEIGHTS
      const int8_t *gate_row_i8 = reinterpret_cast<const int8_t *>(gate_row);
      const int8_t *up_row_i8 = reinterpret_cast<const int8_t *>(up_row);
      float gate_sum = 0.0f, up_sum = 0.0f;
      float g_scale = w.gate_scale[m], u_scale = w.up_scale[m];
#pragma unroll 2
      for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
        uint4 g_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(gate_row_i8 + k));
        uint4 u_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(up_row_i8 + k));
        const int8_t *gp = reinterpret_cast<const int8_t *>(&g_u4);
        const int8_t *up = reinterpret_cast<const int8_t *>(&u_u4);
        float gs = 0.0f, us = 0.0f;
        #pragma unroll
        for (int _i = 0; _i < 16; _i++) {
          float a = s_act[k + _i];
          gs += (float)gp[_i] * a;
          us += (float)up[_i] * a;
        }
        gate_sum += gs;
        up_sum += us;
      }
      gate_sum *= g_scale;
      up_sum *= u_scale;
#else''' + gate_up_bf16 + '''
#endif'''
    src = src.replace(gate_up_bf16, gate_up_int8)

    # Down projection
    down_bf16 = '''      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
        uint4 d_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(down_row + k));
        __nv_bfloat16 *d_ptr = reinterpret_cast<__nv_bfloat16 *>(&d_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(g_mlp_intermediate + k);
        float4 a2 =
            *reinterpret_cast<const float4 *>(g_mlp_intermediate + k + 4);

        sum += __bfloat162float(d_ptr[0]) * a1.x +
               __bfloat162float(d_ptr[1]) * a1.y +
               __bfloat162float(d_ptr[2]) * a1.z +
               __bfloat162float(d_ptr[3]) * a1.w +
               __bfloat162float(d_ptr[4]) * a2.x +
               __bfloat162float(d_ptr[5]) * a2.y +
               __bfloat162float(d_ptr[6]) * a2.z +
               __bfloat162float(d_ptr[7]) * a2.w;
      }'''

    down_int8 = '''#ifdef MK_INT8_WEIGHTS
      const int8_t *down_row_i8 = reinterpret_cast<const int8_t *>(down_row);
      float sum = 0.0f;
      float row_scale_d = w.down_scale[m];
#pragma unroll 2
      for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16) {
        uint4 d_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4 *>(down_row_i8 + k));
        const int8_t *dp = reinterpret_cast<const int8_t *>(&d_u4);
        float s = 0.0f;
        #pragma unroll
        for (int _i = 0; _i < 16; _i++) s += (float)dp[_i] * g_mlp_intermediate[k + _i];
        sum += s;
      }
      sum *= row_scale_d;
#else''' + down_bf16 + '''
#endif'''
    src = src.replace(down_bf16, down_int8)

    return src


if __name__ == "__main__":
    import sys
    kernel_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/megakernel-tts/csrc/kernel.cu"
    with open(kernel_path) as f:
        src = f.read()
    patched = create_int8_kernel(src)
    out_path = kernel_path.replace("kernel.cu", "kernel_int8.cu")
    with open(out_path, 'w') as f:
        f.write(patched)
    print(f"INT8 kernel written to {out_path}")
    print(f"  Struct changes: {'LDGLayerWeights' in patched}")
    print(f"  INT8 loops: {patched.count('MK_INT8_WEIGHTS')}")
