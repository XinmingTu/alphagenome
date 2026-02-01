#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics as stats
import time

import torch

from alphagenome_pytorch import AlphaGenome


def flash_used(model) -> bool:
    used = False
    for layer in model.transformer_unet.transformer.layers:
        attn = layer[0].block
        if getattr(attn, "last_attn_used_flash", False):
            used = True
        pairwise_wrapper = layer[3]
        if pairwise_wrapper is not None:
            pairwise_attn = pairwise_wrapper.block
            if getattr(pairwise_attn, "last_attn_used_flash", False):
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

    return times, flash_used(model), peak_mem


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention2 vs baseline inference.")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096], help="Sequence lengths to benchmark.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--runs", type=int, default=20, help="Timed runs.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs.")
    parser.add_argument("--measure-mem", action="store_true", help="Report peak CUDA memory.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(0)

    base = AlphaGenome(transformer_kwargs={"use_flash_attn": False})
    flash = AlphaGenome(transformer_kwargs={"use_flash_attn": True})
    flash.load_state_dict(base.state_dict())

    for seq_len in args.seq_lens:
        torch.cuda.empty_cache()
        base_times, base_flash_used, base_mem = bench(
            base, seq_len, args.batch, args.runs, args.warmup, device, args.measure_mem
        )
        torch.cuda.empty_cache()
        flash_times, flash_flash_used, flash_mem = bench(
            flash, seq_len, args.batch, args.runs, args.warmup, device, args.measure_mem
        )

        print(f"\nseq_len={seq_len}, batch={args.batch}")
        if base_times is None:
            print("Baseline: OOM")
            continue
        if flash_times is None:
            print("Flash: OOM")
            continue

        base_sum = summarize(base_times)
        flash_sum = summarize(flash_times)

        print(f"Baseline used flash?: {base_flash_used}")
        print(f"Flash used flash?: {flash_flash_used}")
        print(f"Baseline timing (ms): {base_sum}")
        print(f"Flash timing (ms): {flash_sum}")
        print(f"Speedup (mean): {base_sum['mean_ms'] / flash_sum['mean_ms']:.4f}x")
        if args.measure_mem and base_mem and flash_mem:
            def fmt_mem(mem):
                return {
                    "max_allocated_gb": mem["max_allocated_bytes"] / (1024 ** 3),
                    "max_reserved_gb": mem["max_reserved_bytes"] / (1024 ** 3),
                }
            print(f"Baseline peak mem (GB): {fmt_mem(base_mem)}")
            print(f"Flash peak mem (GB): {fmt_mem(flash_mem)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
