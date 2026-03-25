"""
Experiment: KV Cache with PagedAttention vs PagedAttention + TurboQuant
========================================================================
Compares latency, memory usage, and attention accuracy across different
token sequence lengths.

Simulates:
  - PagedAttention KV Cache (FP16 baseline)
  - PagedAttention KV Cache + TurboQuant compression (2-bit, 3-bit, 4-bit)
"""

import numpy as np
import time
import json
import os

from turboquant import TurboQuantProd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulated PagedAttention KV Cache
# ---------------------------------------------------------------------------

class PagedKVCache:
    """Simulates a paged KV cache storing FP16 key/value vectors."""

    def __init__(self, num_heads=32, head_dim=128, page_size=16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.pages_k = []  # list of arrays (page_size, head_dim)
        self.pages_v = []
        self.num_tokens = 0

    def append(self, keys, values):
        """Append tokens. keys/values shape: (num_tokens, num_heads, head_dim)"""
        n = keys.shape[0]
        for i in range(n):
            page_idx = self.num_tokens // self.page_size
            slot = self.num_tokens % self.page_size
            if slot == 0:
                self.pages_k.append(np.zeros((self.page_size, self.num_heads, self.head_dim), dtype=np.float16))
                self.pages_v.append(np.zeros((self.page_size, self.num_heads, self.head_dim), dtype=np.float16))
            self.pages_k[page_idx][slot] = keys[i].astype(np.float16)
            self.pages_v[page_idx][slot] = values[i].astype(np.float16)
            self.num_tokens += 1

    def get_all(self):
        """Return all stored K, V as contiguous arrays."""
        if self.num_tokens == 0:
            return np.empty((0, self.num_heads, self.head_dim)), np.empty((0, self.num_heads, self.head_dim))
        all_k = np.concatenate(self.pages_k, axis=0)[:self.num_tokens]
        all_v = np.concatenate(self.pages_v, axis=0)[:self.num_tokens]
        return all_k.astype(np.float32), all_v.astype(np.float32)

    def memory_bytes(self):
        # K + V, each stored as FP16 (2 bytes per element)
        return self.num_tokens * self.num_heads * self.head_dim * 2 * 2


class PagedKVCacheTurboQuant:
    """Simulates PagedAttention KV cache with TurboQuant compression."""

    def __init__(self, num_heads=32, head_dim=128, page_size=16, bits=3):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.bits = bits
        # One quantizer per head (shared across all tokens)
        self.quantizers_k = [TurboQuantProd(head_dim, bits, seed=42 + h) for h in range(num_heads)]
        self.quantizers_v = [TurboQuantProd(head_dim, bits, seed=1000 + h) for h in range(num_heads)]
        # Store quantized representations
        self.stored_k = []  # list of per-head quantized data
        self.stored_v = []
        self.num_tokens = 0

    def append(self, keys, values):
        """Append tokens with quantization. keys/values: (num_tokens, num_heads, head_dim)"""
        n = keys.shape[0]
        for i in range(n):
            token_k = []
            token_v = []
            for h in range(self.num_heads):
                k_vec = keys[i, h]
                v_vec = values[i, h]
                # Quantize
                idx_k, signs_k, norms_k = self.quantizers_k[h].quantize(k_vec)
                idx_v, signs_v, norms_v = self.quantizers_v[h].quantize(v_vec)
                token_k.append((idx_k, signs_k, norms_k))
                token_v.append((idx_v, signs_v, norms_v))
            self.stored_k.append(token_k)
            self.stored_v.append(token_v)
            self.num_tokens += 1

    def get_all(self):
        """Dequantize and return all K, V."""
        if self.num_tokens == 0:
            return np.empty((0, self.num_heads, self.head_dim)), np.empty((0, self.num_heads, self.head_dim))
        all_k = np.zeros((self.num_tokens, self.num_heads, self.head_dim), dtype=np.float32)
        all_v = np.zeros((self.num_tokens, self.num_heads, self.head_dim), dtype=np.float32)
        for t in range(self.num_tokens):
            for h in range(self.num_heads):
                idx_k, signs_k, norms_k = self.stored_k[t][h]
                idx_v, signs_v, norms_v = self.stored_v[t][h]
                all_k[t, h] = self.quantizers_k[h].dequantize(idx_k, signs_k, norms_k)
                all_v[t, h] = self.quantizers_v[h].dequantize(idx_v, signs_v, norms_v)
        return all_k, all_v

    def memory_bytes(self):
        # K + V, each quantized to self.bits per coordinate + float32 norm per head
        quant_bytes = self.num_tokens * self.num_heads * self.head_dim * self.bits / 8 * 2
        norm_bytes = self.num_tokens * self.num_heads * 4 * 2  # float32 norms for K and V
        return quant_bytes + norm_bytes


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------

def compute_attention(query, keys, values):
    """
    Single-head attention: query (head_dim,), keys (T, head_dim), values (T, head_dim).
    Returns attention output and attention weights.
    """
    d_k = query.shape[0]
    scores = keys @ query / np.sqrt(d_k)  # (T,)
    # Softmax
    scores_max = scores.max()
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum()
    output = attn_weights @ values  # (head_dim,)
    return output, attn_weights


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_experiment(num_tokens, num_heads=8, head_dim=128, bits_list=[2, 3, 4]):
    """Run a single experiment for a given token count."""
    rng = np.random.RandomState(42)

    # Generate random KV pairs (simulating transformer hidden states)
    keys = rng.randn(num_tokens, num_heads, head_dim).astype(np.float32) * 0.1
    values = rng.randn(num_tokens, num_heads, head_dim).astype(np.float32) * 0.1
    query = rng.randn(num_heads, head_dim).astype(np.float32) * 0.1

    results = {}

    # --- Baseline: PagedAttention (FP16) ---
    cache_fp16 = PagedKVCache(num_heads=num_heads, head_dim=head_dim)

    t0 = time.perf_counter()
    cache_fp16.append(keys, values)
    insert_time_fp16 = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_all, v_all = cache_fp16.get_all()
    # Compute attention for each head
    baseline_outputs = []
    baseline_weights = []
    for h in range(num_heads):
        out, w = compute_attention(query[h], k_all[:, h, :], v_all[:, h, :])
        baseline_outputs.append(out)
        baseline_weights.append(w)
    attn_time_fp16 = time.perf_counter() - t0

    results['fp16'] = {
        'insert_latency_ms': insert_time_fp16 * 1000,
        'attention_latency_ms': attn_time_fp16 * 1000,
        'total_latency_ms': (insert_time_fp16 + attn_time_fp16) * 1000,
        'memory_mb': cache_fp16.memory_bytes() / (1024 * 1024),
    }

    # --- TurboQuant variants ---
    for bits in bits_list:
        cache_tq = PagedKVCacheTurboQuant(num_heads=num_heads, head_dim=head_dim, bits=bits)

        t0 = time.perf_counter()
        cache_tq.append(keys, values)
        insert_time_tq = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_tq, v_tq = cache_tq.get_all()
        tq_outputs = []
        tq_weights = []
        for h in range(num_heads):
            out, w = compute_attention(query[h], k_tq[:, h, :], v_tq[:, h, :])
            tq_outputs.append(out)
            tq_weights.append(w)
        attn_time_tq = time.perf_counter() - t0

        # Accuracy metrics
        cosine_sims = []
        weight_kls = []
        max_attn_diffs = []
        for h in range(num_heads):
            # Output cosine similarity
            cos = np.dot(baseline_outputs[h], tq_outputs[h]) / (
                np.linalg.norm(baseline_outputs[h]) * np.linalg.norm(tq_outputs[h]) + 1e-15)
            cosine_sims.append(cos)

            # Attention weight divergence (KL)
            p = baseline_weights[h] + 1e-12
            q_w = tq_weights[h] + 1e-12
            kl = np.sum(p * np.log(p / q_w))
            weight_kls.append(kl)

            # Max absolute attention weight difference
            max_attn_diffs.append(np.max(np.abs(baseline_weights[h] - tq_weights[h])))

        results[f'tq_{bits}bit'] = {
            'insert_latency_ms': insert_time_tq * 1000,
            'attention_latency_ms': attn_time_tq * 1000,
            'total_latency_ms': (insert_time_tq + attn_time_tq) * 1000,
            'memory_mb': cache_tq.memory_bytes() / (1024 * 1024),
            'avg_cosine_sim': float(np.mean(cosine_sims)),
            'avg_kl_divergence': float(np.mean(weight_kls)),
            'avg_max_attn_diff': float(np.mean(max_attn_diffs)),
            'memory_savings_pct': (1 - cache_tq.memory_bytes() / cache_fp16.memory_bytes()) * 100,
        }

    return results


def main():
    print("=" * 70)
    print("  KV Cache: PagedAttention vs PagedAttention + TurboQuant")
    print("=" * 70)

    token_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    bits_list = [2, 3, 4]
    num_heads = 8
    head_dim = 128

    all_results = {}

    for n_tokens in token_sizes:
        print(f"\n{'─' * 60}")
        print(f"  Tokens: {n_tokens}")
        print(f"{'─' * 60}")

        results = run_experiment(n_tokens, num_heads=num_heads, head_dim=head_dim, bits_list=bits_list)
        all_results[n_tokens] = results

        # Print summary
        fp16 = results['fp16']
        print(f"  FP16 Baseline:")
        print(f"    Insert: {fp16['insert_latency_ms']:.2f}ms | Attention: {fp16['attention_latency_ms']:.2f}ms | "
              f"Total: {fp16['total_latency_ms']:.2f}ms | Memory: {fp16['memory_mb']:.2f}MB")

        for bits in bits_list:
            tq = results[f'tq_{bits}bit']
            print(f"  TurboQuant {bits}-bit:")
            print(f"    Insert: {tq['insert_latency_ms']:.2f}ms | Attention: {tq['attention_latency_ms']:.2f}ms | "
                  f"Total: {tq['total_latency_ms']:.2f}ms | Memory: {tq['memory_mb']:.2f}MB")
            print(f"    Savings: {tq['memory_savings_pct']:.1f}% | Cosine: {tq['avg_cosine_sim']:.6f} | "
                  f"KL: {tq['avg_kl_divergence']:.6f}")

    # Save raw results
    serializable = {}
    for n_tokens, res in all_results.items():
        serializable[str(n_tokens)] = res
    with open(os.path.join(RESULTS_DIR, "paged_attention_experiment.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/paged_attention_experiment.json")
    return all_results


if __name__ == "__main__":
    main()
