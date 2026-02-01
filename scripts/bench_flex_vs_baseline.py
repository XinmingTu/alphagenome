#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics as stats
import time

import torch

from alphagenome_pytorch import AlphaGenome


def flex_used(model) -> bool:
    used = False
    for layer in model.transformer_unet.transformer.layers:
        attn = layer[0].block
        if getattr(attn, "last_attn_used_flex", False):
            used = True
    return used


def summarize(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": stats.mean(times) * 1e3,
        "p50_ms": stats.median(times) * 1e3,
        "p90_ms": stats.quantiles(times, n=10)[8] * 1e3 if len(times) >= 10 else max(times) * 1e3,
    }


def bench(model, seq_len: int, batch: int, runs: int, warmup: int, device: torch.device, measure_mem: bool):
    model = model.to(device).eval()
    x = torch.randint(0, 4, (batch, seq_len), device=device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x, organism_index=0, return_embeds=True)
        torch.cuda.synchronize()
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            torch.cuda.empty_cache()
            return None, False, None
        raise

    if measure_mem:
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x, organism_index=0, return_embeds=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    peak_mem = None
    if measure_mem:
        peak_mem = {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    return times, flex_used(model), peak_mem


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FlexAttention vs baseline inference.")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096], help="Sequence lengths to benchmark.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--runs", type=int, default=20, help="Timed runs.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs.")
    parser.add_argument("--measure-mem", action="store_true", help="Report peak CUDA memory.")
    parser.add_argument(
        "--flex-compatible",
        action="store_true",
        help="Use flex-compatible head dims (dim_head_v <= dim_head_qk, power-of-two).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(0)

    transformer_kwargs = {}
    if args.flex_compatible:
        transformer_kwargs = {
            "dim_head_qk": 128,
            "dim_head_v": 128,
        }

    base = AlphaGenome(transformer_kwargs={**transformer_kwargs, "use_flex_attn": False})
    flex = AlphaGenome(transformer_kwargs={**transformer_kwargs, "use_flex_attn": True})
    flex.load_state_dict(base.state_dict())

    for seq_len in args.seq_lens:
        torch.cuda.empty_cache()
        base_times, base_flex_used, base_mem = bench(
            base, seq_len, args.batch, args.runs, args.warmup, device, args.measure_mem
        )
        torch.cuda.empty_cache()
        flex_times, flex_flex_used, flex_mem = bench(
            flex, seq_len, args.batch, args.runs, args.warmup, device, args.measure_mem
        )

        print(f"\nseq_len={seq_len}, batch={args.batch}")
        if base_times is None:
            print("Baseline: OOM")
            continue
        if flex_times is None:
            print("Flex: OOM")
            continue

        base_sum = summarize(base_times)
        flex_sum = summarize(flex_times)

        print(f"Baseline used flex?: {base_flex_used}")
        print(f"Flex used flex?: {flex_flex_used}")
        print(f"Baseline timing (ms): {base_sum}")
        print(f"Flex timing (ms): {flex_sum}")
        print(f"Speedup (mean): {base_sum['mean_ms'] / flex_sum['mean_ms']:.4f}x")
        if args.measure_mem and base_mem and flex_mem:
            def fmt_mem(mem):
                return {
                    "max_allocated_gb": mem["max_allocated_bytes"] / (1024 ** 3),
                    "max_reserved_gb": mem["max_reserved_bytes"] / (1024 ** 3),
                }
            print(f"Baseline peak mem (GB): {fmt_mem(base_mem)}")
            print(f"Flex peak mem (GB): {fmt_mem(flex_mem)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
