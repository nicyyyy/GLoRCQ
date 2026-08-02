"""DeepSeek inference-support helpers (both `deepseek_v2` AND `deepseek` v1).

Keeps the shared inference files (moe_block.py, model_builder.py) lean: the
DeepSeek-specific behavior lives here and is called only behind an
`if self._deepseek:` / `if is_deepseek_moe_block(mod):` guard. Runtime behavior
for Qwen/Mixtral/Qwen3 is therefore unaffected — those paths never enter here.

Two DeepSeek families share this module (their block/router/decoder layouts are
identical; only the attention differs — see the graph section at the bottom):
  * `deepseek_v2` — DeepSeek-V2-Lite (arch DeepseekV2ForCausalLM), MLA attention
    (asymmetric KV: key head_dim 192 / value 128, kv_lora_rank 512).
  * `deepseek`    — DeepSeek-MoE-16B / Dai-2024 (arch DeepseekForCausalLM),
    STANDARD symmetric MHA (num_attention_heads == num_key_value_heads,
    head_dim = hidden/heads = 128). No MLA config fields.
Both: 64 routed + 2 shared experts, top-6, first_k_dense_replace=1 (dense layer 0),
moe_intermediate_size 1408, hidden 2048.

Common specifics handled here (identical for v1 and v2):
  * Router is a custom `MoEGate` (returns (idx, weight, aux), not logits) whose
    config (norm_topk_prob, routed_scaling_factor) differs from Qwen. We compute
    routing directly from `gate.weight` so it works on flattened (N, h) inputs.
  * norm_topk_prob is False → the top-k weights are the raw softmax probs scaled
    by routed_scaling_factor (default 1.0; v1's MoEGate has no such attribute →
    getattr default 1.0 keeps it a no-op, matching modeling_deepseek v1 exactly).
  * Plural, ungated `shared_experts` (a single wider MLP), added directly.
  * Deepseek(V2)DecoderLayer expects `mlp(x)` to return a bare tensor, unlike the
    Qwen/Mixtral decoders which unpack `(hidden_states, router_logits)`.
  * The remote modeling (written for tf ~4.36) calls DynamicCache.get_max_length()
    which was removed in tf 4.51 (renamed get_max_cache_shape). We shim it.
"""
import torch
import torch.nn.functional as F


def is_deepseek_block_obj(original_block) -> bool:
    """True if this HF MoE block is a DeepSeek routed-expert block (v1 or v2).
    Class names are `DeepseekMoE` (v1) / `DeepseekV2MoE` (v2), both prefixed
    `Deepseek`."""
    return type(original_block).__name__.startswith("Deepseek")


def is_deepseek_moe_block(mod) -> bool:
    """True only for a routed DeepSeek MoE block (NOT the dense layer-0
    Deepseek(V2)MLP, which has no `.experts`/`.gate`).

    Matches both the v1 `DeepseekMoE` (DeepSeek-MoE-16B) and the v2
    `DeepseekV2MoE` (DeepSeek-V2-Lite) class names; the dense first-layer MLP
    (`DeepseekMLP` / `DeepseekV2MLP`) lacks `.experts`/`.gate` so it is excluded.
    """
    return (
        type(mod).__name__ in ("DeepseekV2MoE", "DeepseekMoE")
        and hasattr(mod, "experts")
        and hasattr(mod, "gate")
    )


def routing_config(original_block) -> dict:
    """Pull routing config off the DeepSeek MoEGate router."""
    gate = original_block.gate
    return {
        "norm_topk_prob": getattr(gate, "norm_topk_prob", False),
        "routed_scaling_factor": getattr(gate, "routed_scaling_factor", 1.0),
    }


def compute_routing(block, hidden_states):
    """DeepSeek MoEGate routing on flattened (N, h) hidden states.

    softmax over ALL experts (fp32) → greedy top-k. If norm_topk_prob: renormalize
    to sum 1; else scale by routed_scaling_factor (V2-Lite path). Computed from
    gate.weight directly (the MoEGate.forward expects 3-D input, so we can't call
    it on the flattened tensor).

    Returns (router_logits, routing_weights[dtype], selected_experts).
    """
    router_logits = F.linear(hidden_states.float(), block.gate.weight.float(), None)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    else:
        routing_weights = routing_weights * block._routed_scaling_factor
    return router_logits, routing_weights.to(hidden_states.dtype), selected_experts


def add_shared_experts(block, final, hidden_states):
    """Add the ungated plural `shared_experts` MLP output (DeepSeek-V2)."""
    if getattr(block, "shared_experts", None) is not None:
        final = final + block.shared_experts(hidden_states)
    return final


def patch_cache_compat():
    """Add DynamicCache.get_max_length (removed in tf 4.51) so DeepSeek-V2's
    remote modeling generate() works. Idempotent; no-op if already present."""
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


# ---------------------------------------------------------------------------
# bs=1 decode acceleration for DeepSeek-V2-Lite (per-MoE-block CUDA graphs)
# ---------------------------------------------------------------------------
# The shared graph_wrapper captures the WHOLE model as one CUDA graph, but that
# needs a StaticCache + cache_position. DeepSeek-V2's MLA attention decompresses
# and caches asymmetric multi-head K/V (key head_dim=192, value head_dim=128),
# which transformers' StaticCache (single head_dim) cannot hold, and its remote
# modeling uses the legacy DynamicCache API — so full-model capture is not
# available without touching shared code.
#
# Instead we capture ONLY the per-layer MoE block compute (the launch-bound part:
# 64 routed experts + shared expert, dispatched every decode token) into a
# per-block CUDA graph, and leave MLA attention running eager (it is fp16 and
# uses its own DynamicCache). This eliminates the MoE kernel-launch overhead
# while keeping attention correct. Fully DeepSeek-isolated: nothing here runs
# for Qwen/Mixtral, and the shared graph_wrapper is untouched.


class _MoEBlockGraph:
    """Wraps one GraphCompatibleMoeBlock: replays a captured fixed-shape (bs=1,
    seq=1) all-experts graph on decode; falls back to eager for prefill /
    other shapes."""

    def __init__(self, block, hidden_size, device, dtype):
        import torch
        self.block = block
        self.orig_forward = block.forward   # bound method (arch-aware, bare-tensor return)
        self.static_in = torch.zeros((1, 1, hidden_size), device=device, dtype=dtype)
        self.graph = None
        self.static_out = None
        block.graph_mode = True             # fixed-shape all-experts path
        # Warm up on a side stream: builds the lazy vq4 graph cache + cuBLAS/kernel
        # plans so nothing allocates during capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.orig_forward(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self.orig_forward(self.static_in)

    def __call__(self, hidden_states, *args, **kwargs):
        # Decode step: (1, 1, hidden) → replay. Anything else (prefill) → eager.
        if (hidden_states.dim() == 3 and hidden_states.shape[0] == 1
                and hidden_states.shape[1] == 1 and not args and not kwargs):
            self.static_in.copy_(hidden_states)
            self.graph.replay()
            return self.static_out
        self.block.graph_mode = False
        out = self.orig_forward(hidden_states, *args, **kwargs)
        self.block.graph_mode = True
        return out


def install_moe_block_graphs(model, device="cuda:0"):
    """Capture a per-block CUDA graph for every DeepSeek MoE block so bs=1 decode
    replays the expert dispatch instead of launching it. Returns the list of
    runners (keep the reference alive; the graphs own their captured buffers).

    No-op-safe: only wraps GraphCompatibleMoeBlock instances (the routed MoE
    blocks). Dense layer-0 and attention are left untouched.
    """
    import torch
    from .moe_block import GraphCompatibleMoeBlock
    hidden = model.config.hidden_size
    dtype = next(model.parameters()).dtype
    runners = []
    n = 0
    for module in model.modules():
        if isinstance(module, GraphCompatibleMoeBlock):
            runner = _MoEBlockGraph(module, hidden, device, dtype)
            module.forward = runner          # nn.Module.__call__ dispatches to instance attr
            runners.append(runner)
            n += 1
    print(f"[deepseek] captured {n} per-MoE-block CUDA graphs (bs=1 decode)", flush=True)
    return runners


# ---------------------------------------------------------------------------
# torch.compile on the eager MLA attention (Inductor fusion, no fixed shapes)
# ---------------------------------------------------------------------------
# The full-model CUDA graph is blocked by MLA's asymmetric-KV / StaticCache. The
# MoE-block graphs above remove the MoE launch overhead, but MLA attention still
# runs eager, capping the win. torch.compile fuses attention's projection GEMMs /
# RoPE / softmax and cuts Python+launch overhead WITHOUT needing StaticCache or
# fixed shapes, so it sidesteps the blocker and composes with the MoE graphs
# (attention = fused Inductor kernels launched eagerly; MoE = separate cudagraph
# replay). We compile ONLY the attention submodule of each layer.
#
# IMPORTANT: use mode="default" (or "max-autotune"), NOT "reduce-overhead" — the
# latter wraps the compiled region in cudagraphs, which would re-hit the
# StaticCache/asymmetric-KV shape problem and/or clash with our MoE cudagraphs.
# Fully DeepSeek-isolated; Qwen/Mixtral never call this.


# ---------------------------------------------------------------------------
# FULL decode-step CUDA graph for DeepSeek-V2-Lite (attention + MoE together)
# ---------------------------------------------------------------------------
# The per-MoE-block graphs above leave MLA attention running eager, which caps
# bs=1 decode at ~17.7 tok/s. This section captures the ENTIRE decode step —
# embed -> 27 decoder layers (each: eager MLA attention + graph_mode MoE) -> norm
# -> lm_head — into ONE CUDA graph, eliminating attention's per-step Python +
# kernel-launch overhead too.
#
# The blocker for a full-model graph was HF's cache: MLA caches asymmetric
# multi-head K/V (key head_dim=192, value head_dim=128) which StaticCache (one
# head_dim) can't hold, and the remote modeling builds masks / DynamicCache with
# control flow that is not capture-safe. We sidestep BOTH by:
#   * bypassing model.forward — a tiny custom forward loops the stock
#     decoder-layer forwards directly (byte-identical numerics), and
#   * feeding the stock eager attention a self-managed fixed-shape cache
#     (_MLAStaticCache) sized for the asymmetric head dims, indexed by a static
#     cache_position, plus a static additive mask built from cache_position.
# The stock DeepseekV2Attention.forward is used UNCHANGED (no attention rewrite),
# so decode output stays byte-identical to the eager path. Fully DeepSeek-gated.


class _MLAStaticCache:
    """Self-managed fixed-shape KV cache for DeepSeek MLA (asymmetric K/V).

    Pre-allocates per-layer key (…, max_seq, q_head_dim) and value
    (…, max_seq, v_head_dim) buffers. Exposes just the two methods the stock
    DeepseekV2Attention.forward calls — `get_usable_length` and `update` — so the
    attention runs unchanged. Two modes:
      * "prefill": update writes the whole prompt at [0:q_len] and returns the
        buffers sliced to the real length (get_usable_length -> 0), so attention
        is standard causal over the prompt (dynamic, run eager, one-time).
      * "decode": update writes one token at the static `cache_position` index
        and returns the FULL max_seq buffers (get_usable_length -> max_seq-1 so
        kv_seq_len == max_seq, fixed) — capture-safe. A mask built from
        cache_position masks the unwritten tail.
    """

    def __init__(self, config, max_seq, num_layers, device, dtype):
        import torch
        self.max_seq = max_seq
        self.mode = "decode"
        nh = config.num_attention_heads
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 192
        v_head_dim = config.v_head_dim                                   # 128
        self.k_cache = [torch.zeros((1, nh, max_seq, q_head_dim), device=device,
                                    dtype=dtype) for _ in range(num_layers)]
        self.v_cache = [torch.zeros((1, nh, max_seq, v_head_dim), device=device,
                                    dtype=dtype) for _ in range(num_layers)]
        # Static write index (decode). Updated in-place before each graph replay.
        self.cache_position = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_(); v.zero_()

    def get_usable_length(self, new_seq_length, layer_idx=0):
        if self.mode == "prefill":
            return 0
        return self.max_seq - new_seq_length  # decode: new=1 -> kv_seq_len=max_seq

    # stock attention also probes these on some code paths; keep them cheap ints
    def get_seq_length(self, layer_idx=0):
        return self.max_seq

    def get_max_length(self):
        return self.max_seq

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        import torch
        if self.mode == "decode":
            self.k_cache[layer_idx].index_copy_(2, self.cache_position, key_states)
            self.v_cache[layer_idx].index_copy_(2, self.cache_position, value_states)
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        # prefill: write [0:q_len], return the real-length view
        q = key_states.shape[2]
        idx = torch.arange(q, device=key_states.device)
        self.k_cache[layer_idx].index_copy_(2, idx, key_states)
        self.v_cache[layer_idx].index_copy_(2, idx, value_states)
        return self.k_cache[layer_idx][:, :, :q], self.v_cache[layer_idx][:, :, :q]


class _MHAStaticCache:
    """Self-managed fixed-shape KV cache for DeepSeek-MoE-16B (v1) STANDARD MHA.

    Symmetric variant of `_MLAStaticCache`: key and value share one head_dim
    (= hidden_size / num_attention_heads, e.g. 2048/16 = 128), so both buffers
    are (1, num_kv_heads, max_seq, head_dim). Exposes exactly the two methods the
    stock `DeepseekAttention.forward` calls — `get_usable_length` and `update` —
    so that attention runs UNCHANGED (decode output byte-identical to eager). The
    prefill/decode mode semantics are identical to `_MLAStaticCache`:
      * "prefill": get_usable_length -> 0, update writes [0:q_len] and returns the
        real-length view (standard causal attention over the prompt).
      * "decode":  get_usable_length -> max_seq - new_seq_length so kv_seq_len ==
        max_seq (fixed, capture-safe); update writes one token at the static
        cache_position and returns the FULL max_seq buffers. A static additive
        mask (built from cache_position by the graph) masks the unwritten tail.

    This is why DeepSeek-MoE-16B does NOT need the MLA asymmetric-KV hack: plain
    symmetric MHA fits a single-head_dim static cache. (The remaining blocker to
    the *shared* graph_wrapper StaticCache path is unrelated to dims — the tf-4.36
    remote modeling has no `cache_position` kwarg and calls `get_usable_length`
    instead of the StaticCache API — so we still bypass model.forward and drive
    the stock decoder layers directly, exactly as the v2 path does.)
    """

    def __init__(self, config, max_seq, num_layers, device, dtype):
        import torch
        self.max_seq = max_seq
        self.mode = "decode"
        nkv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads   # 128
        self.k_cache = [torch.zeros((1, nkv, max_seq, head_dim), device=device,
                                    dtype=dtype) for _ in range(num_layers)]
        self.v_cache = [torch.zeros((1, nkv, max_seq, head_dim), device=device,
                                    dtype=dtype) for _ in range(num_layers)]
        self.cache_position = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_(); v.zero_()

    def get_usable_length(self, new_seq_length, layer_idx=0):
        if self.mode == "prefill":
            return 0
        return self.max_seq - new_seq_length  # decode: new=1 -> kv_seq_len=max_seq

    def get_seq_length(self, layer_idx=0):
        return self.max_seq

    def get_max_length(self):
        return self.max_seq

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        import torch
        if self.mode == "decode":
            self.k_cache[layer_idx].index_copy_(2, self.cache_position, key_states)
            self.v_cache[layer_idx].index_copy_(2, self.cache_position, value_states)
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        q = key_states.shape[2]
        idx = torch.arange(q, device=key_states.device)
        self.k_cache[layer_idx].index_copy_(2, idx, key_states)
        self.v_cache[layer_idx].index_copy_(2, idx, value_states)
        return self.k_cache[layer_idx][:, :, :q], self.v_cache[layer_idx][:, :, :q]


def _is_mla_config(config) -> bool:
    """True for DeepSeek-V2 MLA attention (asymmetric KV), False for v1 MHA.

    V2 configs carry `qk_nope_head_dim` / `v_head_dim` / `kv_lora_rank`; the v1
    DeepSeek-MoE-16B config has none of these (plain symmetric MHA)."""
    return getattr(config, "qk_nope_head_dim", None) is not None


def _make_deepseek_static_cache(config, max_seq, num_layers, device, dtype):
    """Pick the fixed-shape KV cache matching the attention type: `_MLAStaticCache`
    for DeepSeek-V2 (asymmetric MLA KV), `_MHAStaticCache` for DeepSeek-MoE-16B
    (symmetric MHA)."""
    if _is_mla_config(config):
        return _MLAStaticCache(config, max_seq, num_layers, device, dtype)
    return _MHAStaticCache(config, max_seq, num_layers, device, dtype)


class DeepseekFullDecodeGraph:
    """Prefill (eager) + full decode-step CUDA-graph generator for DeepSeek
    (both v1 MHA and v2 MLA).

    Captures embed -> layers -> norm -> lm_head for a single (bs=1, seq=1) decode
    token into one CUDA graph. The KV cache is chosen by attention type
    (`_MHAStaticCache` for v1 symmetric MHA, `_MLAStaticCache` for v2 MLA) so the
    stock attention.forward is included in the graph, unchanged. This is the
    "full-model decode graph" for DeepSeek-MoE-16B — the analogue of the Qwen
    graph_wrapper path, but self-managed because the tf-4.36 remote modeling is
    not StaticCache/cache_position compatible.
    """

    def __init__(self, model, max_seq_len=384, device="cuda:0"):
        import torch
        from .moe_block import GraphCompatibleMoeBlock
        self.model = model
        self.base = model.model            # Deepseek(V2)Model
        self.lm_head = model.lm_head
        self.layers = self.base.layers
        self.embed = self.base.embed_tokens
        self.norm = self.base.norm
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.max_seq = max_seq_len
        cfg = model.config
        self.min_val = torch.finfo(self.dtype).min
        self.cache = _make_deepseek_static_cache(cfg, max_seq_len, len(self.layers),
                                                 device, self.dtype)
        self._moe_blocks = [m for m in model.modules()
                            if isinstance(m, GraphCompatibleMoeBlock)]
        # Static decode buffers.
        self.static_input_id = torch.ones((1, 1), dtype=torch.long, device=device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._key_pos = torch.arange(max_seq_len, device=device).view(1, 1, 1, -1)
        self.graph = None
        self.static_logits = None

    def _set_moe_graph_mode(self, mode):
        for m in self._moe_blocks:
            m.graph_mode = mode

    def _decode_forward(self):
        """One captured decode step. Reads static_input_id / static_position_ids /
        cache.cache_position; writes into the KV cache; returns logits (1,1,V)."""
        import torch
        h = self.embed(self.static_input_id)                    # (1,1,H)
        # additive mask: key positions > current cache_position are unwritten.
        mask = torch.where(self._key_pos <= self.cache.cache_position.view(1, 1, 1, 1),
                           0.0, self.min_val).to(self.dtype)     # (1,1,1,max_seq)
        for layer in self.layers:
            h = layer(h, attention_mask=mask,
                      position_ids=self.static_position_ids,
                      past_key_value=self.cache, use_cache=True)[0]
        h = self.norm(h)
        return self.lm_head(h)

    def capture(self):
        import torch, gc
        self.cache.mode = "decode"
        self._set_moe_graph_mode(True)
        # Prime cache_position to a mid-window value so warmup/capture exercise the
        # steady-state shapes (mask, index_copy) exactly as replay will.
        self.cache.cache_position.fill_(1)
        self.static_position_ids.fill_(1)
        # Warm up on a side stream: builds lazy vq4 graph caches + cuBLAS plans so
        # nothing allocates during capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self._decode_forward()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._decode_forward()
        print(f"[deepseek] captured full decode-step CUDA graph "
              f"(max_seq={self.max_seq}, layers={len(self.layers)})", flush=True)

    @torch.no_grad()
    def _prefill(self, input_ids):
        import torch
        self.cache.mode = "prefill"
        self.cache.reset()
        self._set_moe_graph_mode(False)
        seq = input_ids.shape[1]
        pos = torch.arange(seq, device=self.device).unsqueeze(0)
        h = self.embed(input_ids)
        mask = torch.full((seq, seq), self.min_val, device=self.device,
                          dtype=self.dtype).triu(1).view(1, 1, seq, seq)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_ids=pos,
                      past_key_value=self.cache, use_cache=True)[0]
        h = self.norm(h[:, -1:])
        logits = self.lm_head(h)
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128):
        import torch
        seq = input_ids.shape[1]
        out = input_ids.clone()
        next_tok = self._prefill(input_ids)          # writes cache[0:seq]
        out = torch.cat([out, next_tok], dim=1)
        if self.graph is None:
            self.capture()
        self.cache.mode = "decode"
        self._set_moe_graph_mode(True)
        for i in range(max_new_tokens - 1):
            pos = seq + i
            if pos >= self.max_seq:
                break
            self.static_input_id.copy_(next_tok)
            self.static_position_ids.fill_(pos)
            self.cache.cache_position.fill_(pos)
            self.graph.replay()
            next_tok = torch.argmax(self.static_logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
        return out


def install_full_decode_graph(model, max_seq_len=384, device="cuda:0"):
    """Build a DeepseekFullDecodeGraph (attention + MoE captured together). Returns
    the runner; call runner.generate(input_ids, max_new_tokens=...).

    Works for both DeepSeek families: the runner auto-selects `_MHAStaticCache`
    for DeepSeek-MoE-16B (v1, symmetric MHA) or `_MLAStaticCache` for
    DeepSeek-V2-Lite (v2, MLA) from model.config. For v1 (plain MHA + many small
    experts, like Qwen) this full-model graph is expected to give a larger bs=1
    decode win than the v2 MLA case; the actual speedup MUST be confirmed on GPU."""
    return DeepseekFullDecodeGraph(model, max_seq_len=max_seq_len, device=device)


def enable_dynamo_graph_break_logs():
    """Turn on torch._dynamo graph-break + recompile reporting (best-effort)."""
    try:
        import torch._logging as _tl
        _tl.set_logs(graph_breaks=True, recompiles=True)
    except Exception:
        try:
            import torch._dynamo as _dyn
            _dyn.config.verbose = True
        except Exception:
            pass


def dynamo_stats():
    """Return a dict of dynamo counters (graph breaks, recompiles, etc.) so the
    bench can quantify how cleanly attention compiled."""
    try:
        import torch._dynamo as _dyn
        return {k: dict(v) for k, v in _dyn.utils.counters.items()}
    except Exception:
        return {}


def install_attention_compile(model, mode="default", dynamic=True,
                              report_graph_breaks=True, cache_size_limit=None):
    """torch.compile each DeepSeek layer's `self_attn` submodule (Inductor
    fusion). Leaves the MoE blocks on their _MoEBlockGraph path untouched, so
    this composes with install_moe_block_graphs(). Returns the number compiled.

    mode: "default" or "max-autotune" (both eager-launch, no cudagraphs). Do NOT
          pass "reduce-overhead" (cudagraphs → re-hits the StaticCache blocker).
    dynamic: True lets Inductor handle varying seq_len (prefill vs decode)
             without recompiling per length.
    cache_size_limit: raise torch._dynamo cache_size_limit. DeepSeek's compiled
             attention guards on len(DynamicCache.key_cache), which grows 0→L
             across layers during prefill — with the default limit (8) this
             overflows and dynamo falls back to eager. Set > num_layers so the
             steady-state decode variant (len==L, constant) actually sticks.

    Compilation is lazy — the first forward triggers it (slow warmup); call a
    warmup generate() before timing.
    """
    import torch
    if mode == "reduce-overhead":
        raise ValueError(
            "reduce-overhead uses cudagraphs → re-hits MLA StaticCache blocker; "
            "use 'default' or 'max-autotune'.")
    if cache_size_limit is not None:
        import torch._dynamo as _dyn
        _dyn.config.cache_size_limit = cache_size_limit
        _dyn.config.accumulated_cache_size_limit = max(
            cache_size_limit, getattr(_dyn.config, 'accumulated_cache_size_limit', 0))
    if report_graph_breaks:
        enable_dynamo_graph_break_logs()
    layers = getattr(model, "model", model).layers
    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        layer.self_attn = torch.compile(attn, mode=mode, dynamic=dynamic)
        n += 1
    print(f"[deepseek] torch.compile applied to {n} attention modules "
          f"(mode={mode}, dynamic={dynamic})", flush=True)
    return n
