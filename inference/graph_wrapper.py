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

        # Static KV Cache
        self.static_cache = StaticCache(
            config=model.config,
            max_batch_size=self.max_batch_size,
            max_cache_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        # Static input buffers (fixed shape for graph)
        self.static_input_ids = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=self.device
        )
        self.static_cache_position = torch.zeros(
            (1,), dtype=torch.long, device=self.device
        )

        self.static_logits = None
        self.graph = None
        self.graph_stream = torch.cuda.Stream()

    def capture_graph(self):
        """
        Capture CUDA Graph for decode phase.

        3-stage process:
          Step 0: Force memory allocation (trigger StaticCache lazy init)
          Step 1: Warmup (3 rounds to stabilize kernel selection)
          Step 2: Record graph
        """
        print(f"[GLoRCQ Graph] Capturing CUDA Graph "
              f"(batch_size={self.max_batch_size}) ...")
        self.model.eval()

        # Step 0: Force memory allocation
        with torch.no_grad():
            self.static_input_ids.fill_(1)
            self.static_cache_position.fill_(0)
            self.model(
                input_ids=self.static_input_ids,
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
                self.model(
                    input_ids=self.static_input_ids,
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

        with torch.cuda.graph(self.graph, stream=self.graph_stream):
            logits = self.model(
                input_ids=self.static_input_ids,
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
                        prompt_len=128, gen_len=128):
    """
    Run speed benchmark comparing standard generation vs CUDA Graph.

    Args:
        model: loaded GLoRCQ model
        tokenizer: HF tokenizer
        max_batch_size: batch size for graph capture
        max_seq_len: maximum sequence length
        prompt_len: length of random prompt
        gen_len: number of tokens to generate
    """
    import random
    import string

    device = next(model.parameters()).device

    # Create random prompt
    alphabet = string.ascii_letters + string.digits + ' '
    prompt = ''.join(random.choice(alphabet) for _ in range(prompt_len))
    inputs = tokenizer(prompt, return_tensors="pt", max_length=prompt_len,
                       truncation=True).to(device)

    print(f"\n{'='*50}")
    print(f"  GLoRCQ Speed Benchmark")
    print(f"  Prompt tokens: {inputs.input_ids.shape[1]}")
    print(f"  Generate: {gen_len} tokens")
    print(f"  Batch size: {max_batch_size}")
    print(f"{'='*50}")

    # Standard generation (no graph)
    print("\n[1/2] Standard generation (no CUDA Graph) ...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        std_out = model.generate(
            inputs.input_ids,
            max_new_tokens=gen_len,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t_std = time.time() - t0
    std_tps = gen_len / t_std

    print(f"  Time: {t_std:.3f}s, Speed: {std_tps:.1f} tok/s")

    # CUDA Graph generation
    print("\n[2/2] CUDA Graph generation ...")
    wrapper = GLoRCQGraphWrapper(model, max_batch_size=max_batch_size,
                                  max_seq_len=max_seq_len)
    wrapper.capture_graph()

    torch.cuda.synchronize()
    graph_out, t_graph = wrapper.generate(inputs.input_ids,
                                           max_new_tokens=gen_len)
    graph_tps = gen_len / t_graph

    print(f"  Time: {t_graph:.3f}s, Speed: {graph_tps:.1f} tok/s")

    # Summary
    speedup = graph_tps / std_tps if std_tps > 0 else 0
    print(f"\n{'='*50}")
    print(f"  Standard: {std_tps:.1f} tok/s")
    print(f"  Graph:    {graph_tps:.1f} tok/s")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"{'='*50}")

    return {
        "standard_tps": std_tps,
        "graph_tps": graph_tps,
        "speedup": speedup,
    }
