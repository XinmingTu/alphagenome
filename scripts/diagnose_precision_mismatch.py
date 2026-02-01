#!/usr/bin/env python3
"""Diagnostic script to find JAX->PyTorch numerical mismatch.

This script compares full model outputs between JAX and PyTorch
in float32 vs bfloat16 to determine if precision is the root cause.

Usage:
    set -a && source .env && set +a && uv run --extra convert python scripts/diagnose_precision_mismatch.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

# Set HuggingFace home directory before imports
os.environ.setdefault("HF_HOME", "/local1/tuxm/huggingface")


def get_best_gpu_index() -> int:
    """Get the GPU index with most available memory.

    If CUDA_VISIBLE_DEVICES is set, returns 0 since only visible GPUs are indexed.
    """
    # If CUDA_VISIBLE_DEVICES is set, use GPU 0 (first visible GPU)
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, using GPU 0")
        return 0

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return 0
        best_idx, best_free = 0, 0
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip())
                if free > best_free:
                    best_idx, best_free = idx, free
        print(f"Auto-selected GPU {best_idx} with {best_free} MiB free memory")
        return best_idx
    except Exception as e:
        print(f"GPU detection failed ({e}), using GPU 0")
        return 0


def compute_comparison_stats(jax_arr: np.ndarray, torch_arr: np.ndarray,
                              atol: float = 1e-2, rtol: float = 1e-2) -> dict[str, Any]:
    """Compute comparison statistics between JAX and PyTorch arrays."""
    jax_flat = jax_arr.flatten().astype(np.float64)
    torch_flat = torch_arr.flatten().astype(np.float64)

    diff = np.abs(jax_flat - torch_flat)
    within_tolerance = diff <= (atol + rtol * np.abs(jax_flat))

    # Pearson correlation
    if len(jax_flat) > 1:
        corr = np.corrcoef(jax_flat, torch_flat)[0, 1]
    else:
        corr = 1.0 if np.allclose(jax_flat, torch_flat) else 0.0

    return {
        'shape': jax_arr.shape,
        'jax_mean': float(np.mean(jax_flat)),
        'jax_std': float(np.std(jax_flat)),
        'torch_mean': float(np.mean(torch_flat)),
        'torch_std': float(np.std(torch_flat)),
        'max_diff': float(np.max(diff)),
        'mean_diff': float(np.mean(diff)),
        'fraction_within_tol': float(np.mean(within_tolerance)),
        'pearson_corr': float(corr),
        'jax_has_nan': bool(np.any(np.isnan(jax_flat))),
        'torch_has_nan': bool(np.any(np.isnan(torch_flat))),
    }


def print_stats(name: str, stats: dict[str, Any]):
    """Pretty print comparison statistics."""
    pct = stats['fraction_within_tol'] * 100
    color = '\033[92m' if pct >= 99 else ('\033[93m' if pct >= 95 else '\033[91m')
    reset = '\033[0m'

    print(f"\n{color}=== {name} ==={reset}")
    print(f"  Shape: {stats['shape']}")
    print(f"  JAX mean/std: {stats['jax_mean']:.6f} / {stats['jax_std']:.6f}")
    print(f"  PyTorch mean/std: {stats['torch_mean']:.6f} / {stats['torch_std']:.6f}")
    print(f"  Max diff: {stats['max_diff']:.6f}, Mean diff: {stats['mean_diff']:.6f}")
    print(f"  {color}Within tolerance: {pct:.2f}%{reset} (target: >=99%)")
    print(f"  Pearson correlation: {stats['pearson_corr']:.6f}")
    if stats['jax_has_nan'] or stats['torch_has_nan']:
        print(f"  WARNING: NaN detected! JAX={stats['jax_has_nan']}, PyTorch={stats['torch_has_nan']}")


def load_jax_model(gpu_idx: int):
    """Load JAX model from HuggingFace."""
    import jax
    from alphagenome_research.model.dna_model import create_from_huggingface

    try:
        gpus = jax.devices("gpu")
        device = gpus[gpu_idx] if gpu_idx < len(gpus) else gpus[0]
        print(f"JAX using GPU device: {device}")
    except RuntimeError:
        device = jax.devices("cpu")[0]
        print(f"JAX using CPU device: {device}")

    model = create_from_huggingface(model_version="all_folds", device=device)
    return model


def load_torch_model(converted_state_dict, gpu_idx: int):
    """Load PyTorch model with converted weights."""
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.alphagenome import set_update_running_var

    model = AlphaGenome()
    model.load_state_dict(converted_state_dict, strict=False)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
        model = model.to(device)
        print(f"PyTorch using GPU device: {device}")
    else:
        print("PyTorch using CPU")

    set_update_running_var(model, False)
    return model


def convert_jax_to_torch_state_dict(jax_model):
    """Convert JAX checkpoint to PyTorch state_dict."""
    from alphagenome_pytorch.convert.convert_checkpoint import (
        convert_checkpoint,
        flatten_nested_dict,
    )

    flat_params = flatten_nested_dict(jax_model._params)
    flat_state = flatten_nested_dict(jax_model._state)
    state_dict = convert_checkpoint(flat_params, flat_state, verbose=False)
    return state_dict


def create_jax_embed_fn(jax_model):
    """Create JAX function that returns embeddings."""
    import haiku as hk
    import jmp
    from alphagenome_research.model import model as model_lib
    from alphagenome_research.model import dna_model as dna_model_lib

    metadata = jax_model._metadata
    model_settings = dna_model_lib.ModelSettings()
    jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, jmp_policy):
            return model_lib.AlphaGenome(
                metadata,
                num_splice_sites=model_settings.num_splice_sites,
                splice_site_threshold=model_settings.splice_site_threshold,
            )(dna_sequence, organism_index)

    def apply_fn(params, state, dna_sequence, organism_index):
        (_, embeddings), _ = _forward.apply(
            params, state, None, dna_sequence, organism_index
        )
        return embeddings

    return apply_fn


def run_comparison(jax_model, torch_model, seq_len: int = 16384):
    """Compare JAX and PyTorch model outputs."""
    import jax.numpy as jnp

    print("\n" + "="*80)
    print(f"MODEL OUTPUT COMPARISON (seq_len={seq_len})")
    print("="*80)

    # Create test input
    np.random.seed(42)
    seq_np = np.random.randint(0, 4, (1, seq_len))

    # JAX setup
    jax_embed_fn = create_jax_embed_fn(jax_model)
    seq_onehot = jnp.array(np.eye(4, dtype=np.float32)[seq_np])
    organism_jax = jnp.zeros((1,), dtype=jnp.int32)

    # JAX forward
    print("\nRunning JAX model (compute=bfloat16)...")
    jax_embeds = jax_embed_fn(
        jax_model._params, jax_model._state, seq_onehot, organism_jax
    )

    jax_outputs = {
        'embeds_1bp': np.asarray(jax_embeds.embeddings_1bp),
        'embeds_128bp': np.asarray(jax_embeds.embeddings_128bp),
        'embeds_pair': np.asarray(jax_embeds.embeddings_pair),
    }

    # PyTorch setup
    device = next(torch_model.parameters()).device
    seq_torch = torch.from_numpy(seq_np).to(device)
    organism_torch = torch.zeros(1, dtype=torch.long, device=device)

    # PyTorch forward (float32)
    print("Running PyTorch model (float32)...")
    with torch.no_grad():
        torch_embeds_fp32 = torch_model(seq_torch, organism_torch, return_embeds=True)

    torch_outputs_fp32 = {
        'embeds_1bp': torch_embeds_fp32[0].cpu().numpy(),
        'embeds_128bp': torch_embeds_fp32[1].cpu().numpy(),
        'embeds_pair': torch_embeds_fp32[2].cpu().numpy(),
    }

    # PyTorch forward (bfloat16)
    print("Running PyTorch model (bfloat16)...")
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            torch_embeds_bf16 = torch_model(seq_torch, organism_torch, return_embeds=True)

    torch_outputs_bf16 = {
        'embeds_1bp': torch_embeds_bf16[0].float().cpu().numpy(),
        'embeds_128bp': torch_embeds_bf16[1].float().cpu().numpy(),
        'embeds_pair': torch_embeds_bf16[2].float().cpu().numpy(),
    }

    # Compare
    print("\n" + "-"*80)
    print("FLOAT32 COMPARISON (current behavior)")
    print("-"*80)
    fp32_results = {}
    for name in ['embeds_1bp', 'embeds_128bp', 'embeds_pair']:
        stats = compute_comparison_stats(jax_outputs[name], torch_outputs_fp32[name])
        print_stats(name, stats)
        fp32_results[name] = stats['fraction_within_tol']

    print("\n" + "-"*80)
    print("BFLOAT16 COMPARISON (matching JAX precision)")
    print("-"*80)
    bf16_results = {}
    for name in ['embeds_1bp', 'embeds_128bp', 'embeds_pair']:
        stats = compute_comparison_stats(jax_outputs[name], torch_outputs_bf16[name])
        print_stats(name, stats)
        bf16_results[name] = stats['fraction_within_tol']

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nFloat32 results (current):")
    for name, pct in fp32_results.items():
        status = "PASS" if pct >= 0.99 else "FAIL"
        print(f"  {name}: {pct*100:.2f}% [{status}]")

    print("\nBFloat16 results (matching JAX):")
    for name, pct in bf16_results.items():
        status = "PASS" if pct >= 0.99 else "FAIL"
        print(f"  {name}: {pct*100:.2f}% [{status}]")

    # Recommendation
    print("\n" + "-"*80)
    bf16_better = all(bf16_results[k] > fp32_results[k] for k in bf16_results)
    bf16_passes = all(bf16_results[k] >= 0.99 for k in bf16_results)

    if bf16_passes:
        print("RECOMMENDATION: Run PyTorch in bfloat16 mode to match JAX precision.")
        print("The tests should pass with torch.autocast('cuda', dtype=torch.bfloat16)")
    elif bf16_better:
        print("FINDING: BFloat16 improves alignment but doesn't fully fix the issue.")
        print("Additional investigation needed for remaining mismatch.")
    else:
        print("FINDING: Precision alone is not the root cause.")
        print("Need to investigate other sources of mismatch.")

    return fp32_results, bf16_results


def main():
    print("="*80)
    print("JAX -> PyTorch Numerical Mismatch Diagnostic")
    print("="*80)

    # Use separate GPUs if available to avoid memory issues
    jax_gpu = 1  # GPU for JAX
    torch_gpu = 2  # GPU for PyTorch

    # If CUDA_VISIBLE_DEVICES is set, use GPU 0 for both
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        jax_gpu = 0
        torch_gpu = 0
        print(f"CUDA_VISIBLE_DEVICES set, using GPU 0 for both models")

    # Load models
    print(f"\nLoading JAX model on GPU {jax_gpu}...")
    jax_model = load_jax_model(jax_gpu)

    print("\nConverting JAX weights to PyTorch...")
    state_dict = convert_jax_to_torch_state_dict(jax_model)

    print(f"\nLoading PyTorch model on GPU {torch_gpu}...")
    torch_model = load_torch_model(state_dict, torch_gpu)

    # Run comparison
    run_comparison(jax_model, torch_model, seq_len=4096)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
