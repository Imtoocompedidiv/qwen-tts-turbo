"""HONEST TTFP measurement. No lies this time.

Measures:
1. GPU TTFP with CUDA events (real GPU compute from request to first frame READY)
2. CPU queue TTFP (what the server currently reports - potentially misleading)
3. Full generation: real frames/sec throughput
4. Audio quality: generates actual audio samples
"""
import subprocess, sys, os, time, re, json, shutil, signal, math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

LOG = "/workspace/bench_output.txt"
RESULT = "/workspace/result.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")
def run(cmd, **kw):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.stdout.strip():
        for l in r.stdout.strip().split("\n")[-5:]: log(f"  {l}")
    if r.returncode != 0 and r.stderr.strip():
        for l in r.stderr.strip().split("\n")[-5:]: log(f"  ERR: {l}")
    return r

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        if self.path=="/log":
            self.send_header("Content-Type","text/plain"); self.end_headers()
            try:
                with open(LOG) as f: self.wfile.write(f.read().encode())
            except: self.wfile.write(b"")
        elif self.path=="/result":
            self.send_header("Content-Type","application/json"); self.end_headers()
            try:
                with open(RESULT) as f: self.wfile.write(f.read().encode())
            except: self.wfile.write(b'{"status":"running"}')
        elif self.path=="/status":
            self.send_header("Content-Type","text/plain"); self.end_headers()
            self.wfile.write(("done" if os.path.exists(RESULT) else "running").encode())
        elif self.path.startswith("/audio/"):
            fname = self.path.split("/")[-1]
            fpath = f"/workspace/{fname}"
            if os.path.exists(fpath):
                self.send_header("Content-Type","audio/wav"); self.end_headers()
                with open(fpath,"rb") as f: self.wfile.write(f.read())
            else:
                self.send_header("Content-Type","text/plain"); self.end_headers()
                self.wfile.write(b"not found")
        else:
            self.send_header("Content-Type","text/plain"); self.end_headers()
            self.wfile.write(b"/log /result /status /audio/<name>.wav")
    def log_message(self,*a): pass

def main():
    os.makedirs("/workspace", exist_ok=True)
    with open(LOG,"w") as f: f.write("")
    Thread(target=lambda: HTTPServer(("0.0.0.0",8080),H).serve_forever(), daemon=True).start()
    log("HTTP :8080")
    result = {"status":"running"}

    try:
        for attempt in range(3):
            try: run("pip install -q faster-qwen3-tts soundfile ninja", timeout=180); break
            except Exception as e:
                log(f"  pip attempt {attempt+1}: {e}")
                if attempt == 2: raise
        run('python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\', local_dir=\'/workspace/model\', ignore_patterns=[\'*.md\'])"', timeout=300)
        run("git clone --depth 1 https://github.com/jayanth-kumar-morem/qwen-megakernel-tts /workspace/megakernel-tts 2>/dev/null || true", timeout=60)

        import torch, torch.nn.functional as F, soundfile as sf
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        sys.path.insert(0, "/workspace/megakernel-tts")

        cc = torch.cuda.get_device_capability()
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        arch = f"sm_{cc[0]}{cc[1]}"
        num_blocks = min(sms, 128)
        log(f"GPU: {torch.cuda.get_device_name(0)} {arch} {sms} SMs")
        result["gpu"] = torch.cuda.get_device_name(0)

        # Patch kernels
        csrc = "/workspace/megakernel-tts/csrc"
        kp = os.path.join(csrc, "kernel.cu")
        with open(kp) as f: orig_src = f.read()
        def ns(m):
            i=m.group(1); w=m.group(2).split('\n')[0]
            return f"{i}{w}\n{i}  __nanosleep(100);\n{i}}}"
        pred_src = re.sub(r'^( +)(while \(.+?\) \{\n\s*\})', lambda m: ns(m), orig_src, flags=re.MULTILINE)
        with open(kp, 'w') as f: f.write(pred_src)

        talker_src = pred_src
        for old, new in [
            ('constexpr int HIDDEN_SIZE = 1024;', '#ifndef MK_HIDDEN_SIZE\n#define MK_HIDDEN_SIZE 1024\n#endif\nconstexpr int HIDDEN_SIZE = MK_HIDDEN_SIZE;'),
            ('constexpr int INTERMEDIATE_SIZE = 3072;', '#ifndef MK_INTERMEDIATE_SIZE\n#define MK_INTERMEDIATE_SIZE 3072\n#endif\nconstexpr int INTERMEDIATE_SIZE = MK_INTERMEDIATE_SIZE;'),
            ('constexpr int NUM_Q_HEADS = 16;', '#ifndef MK_NUM_Q_HEADS\n#define MK_NUM_Q_HEADS 16\n#endif\nconstexpr int NUM_Q_HEADS = MK_NUM_Q_HEADS;'),
            ('constexpr int NUM_KV_HEADS = 8;', '#ifndef MK_NUM_KV_HEADS\n#define MK_NUM_KV_HEADS 8\n#endif\nconstexpr int NUM_KV_HEADS = MK_NUM_KV_HEADS;'),
        ]: talker_src = talker_src.replace(old, new)
        with open(os.path.join(csrc, "kernel_talker.cu"), 'w') as f: f.write(talker_src)

        bp = "/workspace/megakernel-tts/qwen_megakernel/build_tts.py"
        with open(bp) as f: bsrc = f.read()
        bsrc = bsrc.replace('"-arch=sm_120a",', f'"-arch={arch}",')
        bsrc = bsrc.replace("_env_int('LDG_NUM_BLOCKS', 128)", f"_env_int('LDG_NUM_BLOCKS', {num_blocks})")
        with open(bp,'w') as f: f.write(bsrc)
        shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions"), ignore_errors=True)

        # Load model
        log("Loading model...")
        from faster_qwen3_tts import FasterQwen3TTS
        from faster_qwen3_tts.sampling import sample_logits
        from safetensors.torch import load_file
        model = FasterQwen3TTS.from_pretrained("/workspace/model")
        for _ in range(3):
            model.generate_custom_voice(text="Test.", language="French", speaker="Vivian", instruct="")

        inner = model.model.model; talker = inner.talker
        ce = talker.get_input_embeddings()
        codec_head = talker.codec_head
        pred_embeds = talker.code_predictor.get_input_embeddings()
        _, _, config, tie, tam, tth_dummy, tpe = model._prepare_generation_custom(
            text="Test.", language="French", speaker="Vivian", instruct="")
        eos_id = config.codec_eos_token_id
        sm_mask = torch.zeros(config.vocab_size, dtype=torch.bool, device="cuda")
        for i in range(max(0, config.vocab_size-1024), config.vocab_size):
            if i != eos_id: sm_mask[i] = True

        # Build predictor megakernel
        log("Building predictor megakernel...")
        from qwen_megakernel.model_tts import CodePredictorKernel
        sd = load_file("/workspace/model/model.safetensors", device="cuda")
        prefix = "talker.code_predictor.model."
        cp = {}
        for k, v in sd.items():
            if k.startswith(prefix): cp[k[len(prefix):]] = v
            elif k.startswith("talker.code_predictor.") and not k.startswith(prefix):
                cp[k[len("talker.code_predictor."):]] = v
        ew = sd["talker.model.codec_embedding.weight"]
        pw = sd["talker.code_predictor.small_to_mtp_projection.weight"]
        pb = sd.get("talker.code_predictor.small_to_mtp_projection.bias")
        mk_pred = CodePredictorKernel({"code_predictor": cp, "embed_weight": ew}, device="cuda")
        def proj(x):
            return F.linear(x.float(), pw.float(), pb.float() if pb is not None else None).bfloat16()
        proj_ew = F.linear(ew.float(), pw.float(), pb.float() if pb is not None else None).to(torch.bfloat16)
        proj_codec = [F.linear(mk_pred.codec_embeddings[g].float(), pw.float(),
                      pb.float() if pb is not None else None).to(torch.bfloat16)
                      for g in range(mk_pred.num_groups)]

        signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(20)
        for _ in range(5):
            mk_pred.reset()
            mk_pred._step_with_embed(proj(torch.randn(2048, dtype=torch.bfloat16, device="cuda")))
        signal.alarm(0)
        log("  OK")

        # Build talker megakernel
        log("Building talker megakernel...")
        from torch.utils.cpp_extension import load as cpp_load
        TL=28; TH=2048; TKV=8; THD=128; TMAX=2048; TINT=6144
        FLAGS = ["-DMK_HIDDEN_SIZE=2048","-DMK_INTERMEDIATE_SIZE=6144","-DMK_NUM_Q_HEADS=16","-DMK_NUM_KV_HEADS=8",
                 f"-DLDG_NUM_BLOCKS={num_blocks}","-DLDG_BLOCK_SIZE=512","-DLDG_LM_NUM_BLOCKS=16","-DLDG_LM_BLOCK_SIZE=384",
                 "-DLDG_LM_ROWS_PER_WARP=2","-DLDG_ATTN_BLOCKS=16","-DLDG_PREFETCH_QK=0","-DLDG_PREFETCH_THREAD_STRIDE=10",
                 "-DLDG_PREFETCH_DOWN=1","-DLDG_PREFETCH_ELEM_STRIDE=1","-DLDG_PREFETCH_BLOCK_STRIDE=1",
                 "-DLDG_PREFETCH_GATE=1","-DLDG_PREFETCH_UP=1","-DLDG_VOCAB_SIZE=3072",
                 "-DLDG_USE_UINT4","-DLDG_ATTENTION_VEC4","-DLDG_WEIGHT_LDCS","-DLDG_MLP_SMEM"]
        CUDA_FLAGS = ["-O3","--use_fast_math","-std=c++17","--expt-relaxed-constexpr",f"-arch={arch}",f"-I{csrc}"] + FLAGS
        cpp_load(name="qwen_mk_talker_honest",
                 sources=[os.path.join(csrc,"torch_bindings.cpp"),os.path.join(csrc,"kernel_talker.cu")],
                 extra_cuda_cflags=CUDA_FLAGS, extra_cflags=[f"-I{csrc}"], verbose=False)

        tlw = []
        for i in range(TL):
            p = f"talker.model.layers.{i}."
            tlw.extend([sd[p+"input_layernorm.weight"].contiguous(),
                sd[p+"self_attn.q_proj.weight"].contiguous(), sd[p+"self_attn.k_proj.weight"].contiguous(),
                sd[p+"self_attn.v_proj.weight"].contiguous(), sd[p+"self_attn.q_norm.weight"].contiguous(),
                sd[p+"self_attn.k_norm.weight"].contiguous(), sd[p+"self_attn.o_proj.weight"].contiguous(),
                sd[p+"post_attention_layernorm.weight"].contiguous(),
                sd[p+"mlp.gate_proj.weight"].contiguous(), sd[p+"mlp.up_proj.weight"].contiguous(),
                sd[p+"mlp.down_proj.weight"].contiguous()])
        tlw_p = torch.empty(len(tlw), dtype=torch.int64, device="cuda")
        for i, w in enumerate(tlw): tlw_p[i] = w.data_ptr()
        tfn = sd["talker.model.norm.weight"].contiguous()

        ROPE_THETA=1e6; MROPE_T=24
        inv_freq_mr = 1.0/(ROPE_THETA**(torch.arange(0,MROPE_T*2,2,dtype=torch.float32)/THD))
        pos_t = torch.arange(TMAX, dtype=torch.float32)
        cos_mr = torch.ones(TMAX, THD); sin_mr = torch.zeros(TMAX, THD)
        for p_i in range(TMAX):
            for i in range(MROPE_T):
                a = pos_t[p_i]*inv_freq_mr[i]
                cos_mr[p_i,i]=torch.cos(a); cos_mr[p_i,i+THD//2]=torch.cos(a)
                sin_mr[p_i,i]=torch.sin(a); sin_mr[p_i,i+THD//2]=torch.sin(a)
        cos_mr = cos_mr.to(torch.bfloat16).cuda().contiguous()
        sin_mr = sin_mr.to(torch.bfloat16).cuda().contiguous()

        f32d=dict(dtype=torch.float32,device="cuda"); bf16d=dict(dtype=torch.bfloat16,device="cuda")
        tk_c=torch.zeros(TL,TKV,TMAX,THD,**bf16d); tv_c=torch.zeros_like(tk_c)
        t_hid=torch.empty(TH,**bf16d); t_act=torch.empty(TH,**f32d)
        t_res=torch.empty(TH,**f32d); t_q=torch.empty(TH,**f32d)
        t_k=torch.empty(TKV*THD,**f32d); t_v=torch.empty(TKV*THD,**f32d)
        t_att=torch.empty(TH,**f32d); t_mlp=torch.empty(TINT,**f32d)
        t_norm=torch.empty(TH,**f32d)
        t_bmax=torch.empty(4096,**f32d); t_bidx=torch.empty(4096,dtype=torch.int32,device="cuda")
        t_out=torch.empty(1,dtype=torch.int32,device="cuda")
        t_de=torch.zeros(3072,TH,**bf16d); t_dl=torch.zeros(3072,TH,**bf16d)
        t_asc=1.0/math.sqrt(THD)
        talker_decode = torch.ops.qwen_mk_talker_honest.decode

        signal.alarm(20)
        for _ in range(5):
            t_hid.copy_(torch.randn(TH,dtype=torch.bfloat16,device="cuda"))
            talker_decode(t_out,-1,t_de,tlw_p,tfn,t_dl,cos_mr,sin_mr,
                tk_c,tv_c,t_hid,t_act,t_res,t_q,t_k,t_v,t_att,t_mlp,t_norm,t_bmax,t_bidx,TL,0,TMAX,t_asc)
        signal.alarm(0)
        tk_c.zero_(); tv_c.zero_()
        log("  OK")

        # ===================================================================
        # HONEST TTFP TEST: simulate exact server flow with CUDA events
        # ===================================================================
        log("\n" + "="*60)
        log("HONEST TTFP VERIFICATION")
        log("="*60)

        with torch.inference_mode():
            out = talker.forward(inputs_embeds=tie, attention_mask=tam, use_cache=True,
                output_hidden_states=True, return_dict=True, trailing_text_hidden=tth_dummy,
                tts_pad_embed=tpe, generation_step=None, past_hidden=None, past_key_values=None)
            cph = out.past_hidden.clone()
            cl = out.logits[:, -1, :].clone()
            pl = model.talker_graph.prefill_kv(out.past_key_values)

            # Simulate: same combo cached, no KV restore needed
            log("\n--- Test 1: TTFP (first frame, same combo = no KV restore) ---")

            ttfp_gpu_times = []
            ttfp_cpu_times = []
            ttfp_sync_times = []

            for trial in range(30):
                torch.cuda.synchronize()

                # CUDA events for honest GPU timing
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_after_yield = torch.cuda.Event(enable_timing=True)
                ev_after_cpu = torch.cuda.Event(enable_timing=True)

                ev_start.record()
                cpu_start = time.perf_counter()

                # 1. Sample first token (contains .item() sync internally? NO - returns GPU tensor)
                token = sample_logits(cl, temperature=0.9, top_k=50, top_p=1.0,
                    do_sample=True, suppress_mask=sm_mask, suppress_tokens=[eos_id])

                # 2. Predictor megakernel
                mk_pred.reset()
                mk_pred._step_with_embed(proj(cph.squeeze(0).squeeze(0)))
                # token.item() HERE forces a CPU-GPU sync!
                tok_val = token.item()
                tb = torch.tensor([tok_val], dtype=torch.long, device="cuda")
                mk_pred._step_with_embed(F.embedding(tb, proj_ew).squeeze(0))
                cbi = []
                for g in range(mk_pred.num_groups):
                    logits = F.linear(mk_pred._norm_out.to(torch.bfloat16).unsqueeze(0), mk_pred.lm_heads[g]).squeeze(0)
                    t = logits.argmax(keepdim=True).long()
                    cbi.append(t.squeeze())
                    if g < mk_pred.num_groups - 1:
                        mk_pred._step_with_embed(F.embedding(t, proj_codec[g]).squeeze(0))
                codebook_ids = torch.stack(cbi)
                all_cb = torch.cat([token.view(1), codebook_ids])

                # === YIELD POINT === (this is where the server yields to WS handler)
                ev_after_yield.record()
                cpu_yield = time.perf_counter()
                cpu_ttfp = (cpu_yield - cpu_start) * 1000  # CPU queue time (what server reports)

                # === CLIENT RECEIVES DATA === (.cpu() forces sync)
                data = all_cb.cpu().numpy()  # THIS is when GPU actually finishes
                ev_after_cpu.record()
                torch.cuda.synchronize()
                cpu_sync = time.perf_counter()

                gpu_ttfp = ev_start.elapsed_time(ev_after_yield)  # GPU time to yield point
                gpu_full = ev_start.elapsed_time(ev_after_cpu)     # GPU time including .cpu()
                cpu_sync_ttfp = (cpu_sync - cpu_start) * 1000     # CPU time including sync

                ttfp_gpu_times.append(gpu_ttfp)
                ttfp_cpu_times.append(cpu_ttfp)
                ttfp_sync_times.append(cpu_sync_ttfp)

            ttfp_gpu_times.sort()
            ttfp_cpu_times.sort()
            ttfp_sync_times.sort()

            log(f"\n  GPU TTFP (CUDA events, to yield):     p50={ttfp_gpu_times[15]:.2f}ms  min={ttfp_gpu_times[0]:.2f}ms")
            log(f"  CPU TTFP (perf_counter, to yield):     p50={ttfp_cpu_times[15]:.2f}ms  min={ttfp_cpu_times[0]:.2f}ms")
            log(f"  REAL TTFP (CPU with sync, incl .cpu): p50={ttfp_sync_times[15]:.2f}ms  min={ttfp_sync_times[0]:.2f}ms")
            log(f"  Δ GPU vs CPU-queue: {ttfp_gpu_times[15] - ttfp_cpu_times[15]:+.2f}ms")
            log(f"  Δ REAL vs GPU: {ttfp_sync_times[15] - ttfp_gpu_times[15]:+.2f}ms")

            result["gpu_ttfp_p50"] = round(ttfp_gpu_times[15], 2)
            result["cpu_ttfp_p50"] = round(ttfp_cpu_times[15], 2)
            result["real_ttfp_p50"] = round(ttfp_sync_times[15], 2)

            if abs(ttfp_cpu_times[15] - ttfp_gpu_times[15]) > 1.0:
                log(f"\n  ⚠ WARNING: CPU queue TTFP differs from GPU TTFP by >{abs(ttfp_cpu_times[15] - ttfp_gpu_times[15]):.1f}ms!")
                log(f"    The server's reported TTFP would be MISLEADING.")
                result["ttfp_honest"] = False
            else:
                log(f"\n  ✓ CPU and GPU TTFP are within 1ms — server report is honest.")
                result["ttfp_honest"] = True

            # --- Test 2: Full step throughput (predictor + talker) ---
            log("\n--- Test 2: Full step throughput (predictor + talker + overhead) ---")

            def full_step():
                # Predictor
                mk_pred.reset()
                mk_pred._step_with_embed(proj(cph.squeeze(0).squeeze(0)))
                tb2 = torch.tensor([token.item()], dtype=torch.long, device="cuda")
                mk_pred._step_with_embed(F.embedding(tb2, proj_ew).squeeze(0))
                cbi2 = []
                for g in range(mk_pred.num_groups):
                    logits2 = F.linear(mk_pred._norm_out.to(torch.bfloat16).unsqueeze(0), mk_pred.lm_heads[g]).squeeze(0)
                    t2 = logits2.argmax(keepdim=True).long()
                    cbi2.append(t2.squeeze())
                    if g < mk_pred.num_groups - 1:
                        mk_pred._step_with_embed(F.embedding(t2, proj_codec[g]).squeeze(0))
                # Overhead
                lih = ce(token.unsqueeze(1))
                chs = [lih]
                for ci in range(config.num_code_groups-1):
                    chs.append(pred_embeds[ci](cbi2[ci].unsqueeze(0).unsqueeze(0)))
                ie = torch.cat(chs, dim=1).sum(1, keepdim=True) + tpe
                # Talker
                t_hid.copy_(ie.view(-1))
                talker_decode(t_out,-1,t_de,tlw_p,tfn,t_dl,cos_mr,sin_mr,
                    tk_c,tv_c,t_hid,t_act,t_res,t_q,t_k,t_v,t_att,t_mlp,t_norm,t_bmax,t_bidx,TL,0,TMAX,t_asc)

            for _ in range(10): full_step()
            step_times = []
            for _ in range(50):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); full_step(); e.record()
                torch.cuda.synchronize()
                step_times.append(s.elapsed_time(e))
            step_times.sort()
            log(f"  Full step (CUDA events): p50={step_times[25]:.2f}ms  min={step_times[0]:.2f}ms")
            log(f"  Throughput: {1000/step_times[25]:.0f} frames/sec = {1000/step_times[25]/12:.1f}x realtime @12Hz")
            result["full_step_p50"] = round(step_times[25], 2)
            result["throughput_fps"] = round(1000/step_times[25])
            result["realtime_factor"] = round(1000/step_times[25]/12, 1)

        # --- Test 3: Generate actual audio ---
        log("\n--- Test 3: Audio generation quality check ---")
        texts = [
            ("Bonjour, ceci est un test de qualité audio avec le megakernel. La voix doit être naturelle et fluide.", "French", "Vivian"),
            ("Hello, this is a quality test with the megakernel. The voice should sound natural and clear.", "English", "Serena"),
        ]
        for text, lang, voice in texts:
            log(f"  Generating [{lang}/{voice}]: {text[:50]}...")
            t0 = time.perf_counter()
            wavs, sr = model.generate_custom_voice(text=text, language=lang, speaker=voice, instruct="")
            gen_time = time.perf_counter() - t0
            dur = len(wavs[0]) / sr
            fname = f"audio_{lang.lower()}.wav"
            sf.write(f"/workspace/{fname}", wavs[0], sr)
            log(f"    {dur:.1f}s audio in {gen_time:.1f}s (RTF={gen_time/dur:.2f})")
            result[f"audio_{lang.lower()}"] = {"duration": round(dur, 1), "gen_time": round(gen_time, 1)}

        log(f"\n  Audio files: /audio/audio_french.wav, /audio/audio_english.wav")

        # --- Summary ---
        log("\n" + "="*60)
        log("FINAL HONEST NUMBERS")
        log("="*60)
        log(f"  GPU:              {result['gpu']}")
        log(f"  GPU TTFP:         {result['gpu_ttfp_p50']}ms (CUDA events, to first frame ready)")
        log(f"  CPU queue TTFP:   {result['cpu_ttfp_p50']}ms (what server reports)")
        log(f"  Real TTFP:        {result['real_ttfp_p50']}ms (including .cpu() sync)")
        log(f"  Full step:        {result['full_step_p50']}ms/frame")
        log(f"  Throughput:       {result['throughput_fps']} fps ({result['realtime_factor']}x realtime)")
        log(f"  TTFP honest:      {result.get('ttfp_honest', '?')}")
        log("="*60)

        result["status"] = "success"

    except TimeoutError:
        signal.alarm(0)
        log("DEADLOCK"); result["status"] = "deadlock"
    except Exception as e:
        log(f"FATAL: {e}")
        import traceback; log(traceback.format_exc())
        result["status"] = "error"; result["error"] = str(e)

    with open(RESULT,"w") as f: json.dump(result, f, indent=2)
    log(f"\n=== DONE ===")
    time.sleep(1800)  # Keep alive for audio download

if __name__ == "__main__":
    main()
