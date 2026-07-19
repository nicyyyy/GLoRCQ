"""
CUDA Graph wrapper for GLoRCQ quantized model inference.

Accelerates the decode phase of autoregressive generation by capturing
the forward pass into a CUDA Graph, eliminating kernel launch overhead.

Reference: FluxBin kernel/graphwrapper.py

MoE handling: All experts participate in the graph (routing weights=0 for
inactive experts masks their output). This keeps the execution path fixed
as required by CUDA Graph, while still capturing expert matmul kernels.
"""

import gc
import os
import time
import torch
import torch.nn as nn
from transformers import StaticCache


class GLoRCQGraphWrapper:
    """
    CUDA Graph accelerated inference for GLoRCQ quantized models.

    Phases:
      - Prefill: standard PyTorch forward (dynamic seq_len, no graph)
      - Decode: CUDA Graph replay (fixed batch x 1 token)

    Usage:
        wrapper = GLoRCQGraphWrapper(model, max_batch_size=1, max_seq_len=4096)
        output_ids, duration = wrapper.generate(input_ids, max_new_tokens=128)
    """

    def __init__(self, model, max_batch_size=1, max_seq_len=4096):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Mixtral defaults `output_router_logits=True`, which returns a list of
        # per-layer tensors during forward — this both inflates memory during
        # graph capture and produces variable-shape outputs that break replay.
        # Force it off before capture. Qwen2Moe defaults False so no-op there.
        model.config.output_router_logits = False
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            try:
                model.generation_config.output_router_logits = False
            except Exception:
                pass

        # Static KV Cache
        self.static_cache = StaticCache(
            config=model.config,
            max_batch_size=self.max_batch_size,
            max_cache_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        # Mixtral's stock attention calls past_key_value.get_usable_length(),
        # whose implementation does `if previous_seq_length + new_seq_length > max_length`
        # where previous_seq_length is a CUDA 0-dim tensor from StaticCache.get_seq_length
        # (default: `(key_cache[layer_idx][0,0].any(dim=-1)).sum()`). The Python `>`
        # forces a `.item()` sync — forbidden during CUDA-Graph capture.
        # Patch get_seq_length to a Python int equal to max_cache_len; this makes
        # the > check take the "clamp" branch and returns max_length - new_seq_length,
        # so kv_seq_len = max_cache_len — sufficient for RoPE size and safe under capture.
        _mcl = max_seq_len
        self.static_cache.get_seq_length = lambda layer_idx=0: _mcl

        # Static input buffers (fixed shape for graph)
        self.static_input_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=self.device
        )
        self.static_cache_position = torch.zeros(
            (1,), dtype=torch.long, device=self.device
        )
        # Mixtral's stock forward derives position_ids from cache_position in a
        # way that misfires under CUDA-Graph capture (indexing cos with a stale
        # tensor → device-side assert at apply_rotary_pos_emb). Passing
        # position_ids explicitly avoids that path. Qwen2Moe accepts and honors
        # position_ids too — safe to pass unconditionally.
        self.static_position_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=self.device
        )

        self.static_logits = None
        self.graph = None
        self.graph_stream = torch.cuda.Stream()

    def _mixtral_gather_available(self):
        """True iff the model's MoE blocks are GraphCompatibleMoeBlock with the
        top-k gather graph path enabled (num_experts <= gather threshold). If a
        different block type is installed (e.g. MixtralFastMoeBlock) or the
        gather gate is off, the all-experts graph would run -> report False so
        capture stays disabled and standard decode is used instead.
        """
        from .moe_block import GraphCompatibleMoeBlock
        found = False
        for module in self.model.modules():
            if isinstance(module, GraphCompatibleMoeBlock):
                if not getattr(module, "_use_gather_graph", False):
                    return False
                found = True
        return found

    def _set_moe_graph_mode(self, mode: bool):
        """Toggle graph_mode on all GraphCompatibleMoeBlock modules.

        Mixed vq4 + fp16-shim expert layers now handled by _forward_graph_vq4's
        _apply_fp16_shim_override — everything runs graph-safe.
        """
        from .moe_block import GraphCompatibleMoeBlock
        for module in self.model.modules():
            if isinstance(module, GraphCompatibleMoeBlock):
                module.graph_mode = mode

    def capture_graph(self):
        """
        Capture CUDA Graph for decode phase.

        3-stage process:
          Step 0: Force memory allocation (trigger StaticCache lazy init)
          Step 1: Warmup (3 rounds to stabilize kernel selection)
          Step 2: Record graph
        """
        # Mixtral auto-disable: the graph decode path runs a STATIC all-experts
        # MoE (moe_block._forward_graph forces E == num_experts). With Mixtral's
        # top_k=2 of 8 experts that is 4x the expert-matmul work of the standard
        # top_k decode loop -> net-negative (measured Mixtral: graph 6.0 tok/s <
        # standard 7.9 tok/s). Skip capture and leave self.graph = None so
        # replay() transparently falls back to the standard forward, and let the
        # caller keep graph_mode = False so that fallback uses the sparse top_k
        # decode path (not the all-experts graph path). Qwen MoE (many small
        # experts, graph is a real speedup) is unaffected.
        # Task #203: a top-k GATHER graph path exists for Mixtral (moe_block
        # _forward_graph_vq4_gather) that computes only top_k experts. It is
        # CORRECT and net-positive, but only marginally (~1.02x) — Mixtral
        # decode is memory-bound on the huge (inter=14336) expert GEMVs, so
        # eliminating launch overhead (all a CUDA graph buys) has almost no
        # headroom. It also shifts the shim-layer down_proj to the fp16 grouped
        # kernel (vs the standard path's fp32-python fallback). So it is left
        # OFF BY DEFAULT: Mixtral runs standard decode (today's numbers) unless
        # explicitly opted in with GLORCQ_MIXTRAL_GRAPH=1. ("0" or unset -> the
        # legacy disabled behavior.) When enabled it still requires the gather
        # path to actually be installed, else the net-negative all-experts
        # graph would run.
        _cfg = getattr(self.model, "config", None)
        _arch = list(getattr(_cfg, "architectures", None) or [])
        _is_mixtral = ("MixtralForCausalLM" in _arch
                       or getattr(_cfg, "model_type", "") == "mixtral")
        if _is_mixtral:
            _env = os.environ.get("GLORCQ_MIXTRAL_GRAPH", "")
            _enable = (_env == "1") and self._mixtral_gather_available()
            if not _enable:
                print("[GLoRCQ Graph] Mixtral detected -> CUDA Graph disabled "
                      "(default; set GLORCQ_MIXTRAL_GRAPH=1 to opt into the "
                      "top-k gather graph path); using standard decode.")
                self.graph = None
                return
            print("[GLoRCQ Graph] Mixtral top-k gather graph path enabled "
                  "(GLORCQ_MIXTRAL_GRAPH=1).")

        # Fp16-shim experts are now handled by _forward_graph_vq4 via a
        # graph-safe overlay path (see _apply_fp16_shim_override in moe_block).
        # No need to skip capture — batched bmm on static indices is compatible
        # with CUDA Graph.
        print(f"[GLoRCQ Graph] Capturing CUDA Graph "
              f"(batch_size={self.max_batch_size}) ...")
        self.model.eval()
        self._set_moe_graph_mode(True)

        # Step 0: Force memory allocation
        with torch.no_grad():
            self.static_input_ids.fill_(1)
            self.static_cache_position.fill_(0)
            self.static_position_ids.fill_(0)
            self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                cache_position=self.static_cache_position,
                past_key_values=self.static_cache,
                use_cache=True,
                return_dict=False,
            )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # Step 1: Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.static_input_ids.fill_(1)
                self.static_cache_position.fill_(10)
                self.static_position_ids.fill_(10)
                self.model(
                    input_ids=self.static_input_ids,
                    position_ids=self.static_position_ids,
                    cache_position=self.static_cache_position,
                    past_key_values=self.static_cache,
                    use_cache=True,
                    return_dict=False,
                )
        torch.cuda.current_stream().wait_stream(s)
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Capture
        self.graph = torch.cuda.CUDAGraph()
        self.static_input_ids.fill_(1)
        self.static_cache_position.fill_(0)
        self.static_position_ids.fill_(0)

        with torch.cuda.graph(self.graph, stream=self.graph_stream):
            logits = self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                cache_position=self.static_cache_position,
                past_key_values=self.static_cache,
                use_cache=True,
                return_dict=False,
            )[0]
            self.static_logits = logits

        print("[GLoRCQ Graph] Capture complete.")

    @torch.no_grad()
    def replay(self, input_ids, cache_position_val):
        """
        Replay captured graph with new inputs.

        Args:
            input_ids: (batch_size, 1) token ids
            cache_position_val: int, current sequence position
        """
        assert input_ids.shape[0] == self.max_batch_size, \
            (f"Input batch size {input_ids.shape[0]} != "
             f"graph batch size {self.max_batch_size}")

        self.static_input_ids.copy_(input_ids)
        self.static_cache_position.fill_(cache_position_val)
        self.static_position_ids.fill_(cache_position_val)
        if self.graph is None:
            # Graph capture was skipped (e.g. fp16_passthrough experts).
            # Fall back to standard model.forward for this decode step.
            out = self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                cache_position=self.static_cache_position,
                past_key_values=self.static_cache,
                use_cache=True,
                return_dict=False,
            )
            return out[0]
        self.graph.replay()
        return self.static_logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100):
        """
        Generate tokens using prefill + CUDA Graph decode.

        Args:
            input_ids: (batch_size, seq_len) prompt token ids
            max_new_tokens: number of tokens to generate

        Returns:
            output_ids: (batch_size, seq_len + max_new_tokens) generated ids
            duration: float, total generation time in seconds
        """
        start_time = time.time()
        batch_size, seq_len = input_ids.shape
        assert batch_size == self.max_batch_size, \
            (f"Input batch_size ({batch_size}) must match "
             f"max_batch_size ({self.max_batch_size})")

        output_ids = input_ids.clone()

        # Phase 1: Prefill (standard forward, no graph)
        self._set_moe_graph_mode(False)
        self.static_cache.reset()
        cache_position = torch.arange(seq_len, device=self.device)

        logits = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
            return_dict=False,
        )[0]

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)

        torch.cuda.synchronize()
        prefill_end = time.time()

        # Phase 2: Decode (CUDA Graph)
        if self.graph is None:
            self.capture_graph()
        # Enable the static all-experts graph path ONLY if a graph was actually
        # captured. When capture was skipped (self.graph is None — e.g. Mixtral
        # auto-disable or fp16-shim experts) keep graph_mode False so the replay
        # fallback runs the standard sparse top_k decode, not the net-negative
        # all-experts path.
        self._set_moe_graph_mode(self.graph is not None)

        for i in range(max_new_tokens - 1):
            current_pos = seq_len + i
            if current_pos >= self.max_seq_len:
                break

            logits = self.replay(next_token, current_pos)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)

        torch.cuda.synchronize()
        end_time = time.time()

        total_duration = end_time - start_time
        prefill_duration = prefill_end - start_time
        decode_duration = end_time - prefill_end

        return output_ids, total_duration


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------
def run_speed_benchmark(model, tokenizer, max_batch_size=1, max_seq_len=2048,
                        prompt_len=128, gen_len=128, skip_standard=False):
    """
    Run speed benchmark comparing standard generation vs CUDA Graph.

    Args:
        model: loaded GLoRCQ model
        tokenizer: HF tokenizer
        max_batch_size: batch size for graph capture
        max_seq_len: maximum sequence length
        prompt_len: length of random prompt
        gen_len: number of tokens to generate
        skip_standard: if True, only run the CUDA-graph path (no standard
            baseline) — faster when only the graph number is needed.
    """
    import random
    import string

    device = next(model.parameters()).device

    # Create random prompt
    alphabet = string.ascii_letters + string.digits + ' '
    prompt = ''.join(random.choice(alphabet) for _ in range(prompt_len))
    inputs = tokenizer(prompt, return_tensors="pt", max_length=prompt_len,
                       truncation=True).to(device)
    # Replicate the prompt across the batch so both the standard and the graph
    # path actually run at max_batch_size (the graph wrapper asserts the input
    # batch == max_batch_size). tok/s below is PER-SEQUENCE (gen_len / time);
    # total throughput = per-seq * batch.
    input_ids = inputs.input_ids
    if max_batch_size > 1:
        input_ids = input_ids.repeat(max_batch_size, 1)
    attn_mask = torch.ones_like(input_ids)

    print(f"\n{'='*50}")
    print(f"  GLoRCQ Speed Benchmark")
    print(f"  Prompt tokens: {input_ids.shape[1]}")
    print(f"  Generate: {gen_len} tokens")
    print(f"  Batch size: {max_batch_size}")
    print(f"{'='*50}")

    # Standard generation (no graph) — skipped in graph-only mode
    std_tps = None
    if not skip_standard:
        print("\n[1/2] Standard generation (no CUDA Graph) ...")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            std_out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=gen_len,
                do_sample=False,
            )
        torch.cuda.synchronize()
        t_std = time.time() - t0
        std_tps = gen_len / t_std
        print(f"  Time: {t_std:.3f}s, Speed: {std_tps:.1f} tok/s (per-seq; "
              f"total {std_tps*max_batch_size:.1f})")
    else:
        print("\n[1/2] Standard generation SKIPPED (graph-only mode)")

    # CUDA Graph generation
    print("\n[2/2] CUDA Graph generation ...")
    wrapper = GLoRCQGraphWrapper(model, max_batch_size=max_batch_size,
                                  max_seq_len=max_seq_len)
    wrapper.capture_graph()

    torch.cuda.synchronize()
    graph_out, t_graph = wrapper.generate(input_ids,
                                           max_new_tokens=gen_len)
    graph_tps = gen_len / t_graph

    print(f"  Time: {t_graph:.3f}s, Speed: {graph_tps:.1f} tok/s (per-seq; "
          f"total {graph_tps*max_batch_size:.1f})")

    # Summary
    speedup = (graph_tps / std_tps) if (std_tps and std_tps > 0) else 0
    print(f"\n{'='*50}")
    if std_tps is not None:
        print(f"  Standard: {std_tps:.1f} tok/s")
        print(f"  Speedup:  {speedup:.2f}x")
    else:
        print(f"  Standard: (skipped)")
    print(f"  Graph:    {graph_tps:.1f} tok/s")
    print(f"{'='*50}")

    return {
        "standard_tps": std_tps,
        "graph_tps": graph_tps,
        "speedup": speedup,
    }
