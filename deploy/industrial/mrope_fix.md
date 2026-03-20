# M-RoPE Fix for Qwen3-TTS Megakernel

## Current Bug
The megakernel uses standard 1D RoPE:
```cuda
const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;
```
This applies the SAME rotary position to ALL 128 dimensions of each head.

## Qwen3-TTS M-RoPE Specification

From `config.json`:
```json
{
  "rope_scaling": {
    "type": "default",
    "mrope_section": [24, 20, 20],
    "interleaved": true
  }
}
```

### What M-RoPE does:
- `head_dim = 128`, so `half_dim = 64`
- `mrope_section = [24, 20, 20]` → splits the 64 half-dimensions into 3 sections
- Each section uses a DIFFERENT position from `position_ids[3, batch, seq]`
- `interleaved = true` means the sections interleave rather than concatenate

### Interleaved M-RoPE mapping:
For dimension index `i` in the head (0-127):
- `half_i = i % 64` (for the cos/sin pair)
- `section_idx = which section half_i falls into`
  - half_i in [0, 24): section 0 → uses position_ids[0]
  - half_i in [24, 44): section 1 → uses position_ids[1]
  - half_i in [44, 64): section 2 → uses position_ids[2]

But with interleaving, the actual index mapping is:
- For each pair (i, i+64):
  - pair_idx = i (0-63)
  - section = 0 if pair_idx < 24, 1 if pair_idx < 44, 2 if pair_idx < 64
  - cos/sin from `cos_table[position_ids[section] * HEAD_DIM + i]`

## Required Kernel Changes

### 1. Function signature change
```cuda
// OLD:
void ldg_attention(..., int position, ...)

// NEW:
void ldg_attention(..., int pos_0, int pos_1, int pos_2, ...)
```

### 2. RoPE application change
```cuda
// OLD (line 346-347):
const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

// For each dimension i, use the same cos_pos[i] and sin_pos[i]

// NEW:
// For K heads (block 0) and Q heads (attention blocks):
for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
    int half_i = i % (HEAD_DIM / 2);  // 0-63
    int section;
    if (half_i < 24) section = 0;
    else if (half_i < 44) section = 1;
    else section = 2;

    int pos = (section == 0) ? pos_0 : (section == 1) ? pos_1 : pos_2;
    float cos_val = __bfloat162float(cos_table[pos * HEAD_DIM + i]);
    float sin_val = __bfloat162float(sin_table[pos * HEAD_DIM + i]);

    // Apply rotary: x_rot = x * cos - x_partner * sin
    // Partner is i+64 if i<64, or i-64 if i>=64
    // ... standard rotary application
}
```

### 3. Propagate through all callers
- `ldg_decode_step` passes position to `ldg_attention`
- `ldg_prefill_step` passes position to `ldg_attention`
- `autoregressive_decode` passes position
- All need to pass `pos_0, pos_1, pos_2` instead of single `position`

### 4. Python side: compute position_ids
The Python wrapper needs to compute the 3 position IDs:
```python
# For decode step at position p:
# Since rope_deltas is typically 0 for TTS decode,
# all 3 positions are usually the same: pos_0 = pos_1 = pos_2 = p
# But the cos/sin tables are PRE-COMPUTED with M-RoPE section mapping,
# so the actual fix might be simpler: use the model's pre-computed
# cos/sin tables which already encode the section mapping.
```

## Key Insight
If the cos/sin tables are pre-computed by the model with M-RoPE already applied,
then the kernel just needs to use the CORRECT position per section.
For TTS decode (where all 3 position_ids are the same), the 1D RoPE might
actually produce the SAME result as M-RoPE — the divergence only appears
when the 3 positions differ (which happens during prefill with mixed modalities).

## Testing Strategy
1. Compare cos/sin values from the model at a given position
2. Compare with the megakernel's cos_table at the same position
3. If they match for decode (position_ids all same), the bug might not affect decode quality
4. The real issue may only be in prefill (which we skip with prefix KV cache)
