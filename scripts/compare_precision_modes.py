#!/usr/bin/env python3
"""Compare JAX (bf16 ground truth) vs PyTorch float32 and AMP bf16.

Usage:
  set -a && source .env && set +a && uv run --extra convert python scripts/compare_precision_modes.py

Env:
  SEQ_LEN: sequence length (default 4096; 16384 may NaN in JAX bf16)
  SEED: RNG seed (default 42)
  JAX_DEVICE: 'gpu' or 'cpu' (default: gpu if available)
  ALPHAGENOME_TORCH_BF16: 1 to enable native bf16 attention path (optional)
"""
from __future__ import annotations

import os
import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/local1/tuxm/huggingface")

SEQ_LEN = int(os.environ.get("SEQ_LEN", "4096"))
SEED = int(os.environ.get("SEED", "42"))
JAX_DEVICE = os.environ.get("JAX_DEVICE", "gpu").lower()

import jax
import jax.numpy as jnp
import haiku as hk
import jmp
from alphagenome_research.model import model as model_lib
from alphagenome_research.model import dna_model as dna_model_lib
from alphagenome_research.model.dna_model import create_from_huggingface
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import set_update_running_var
from alphagenome_pytorch.convert.convert_checkpoint import convert_checkpoint, flatten_nested_dict


def pick_jax_device():
    if JAX_DEVICE == "cpu":
        return jax.devices("cpu")[0]
    try:
        gpus = jax.devices("gpu")
        return gpus[0] if gpus else jax.devices("cpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


def stats_report(name: str, jax_arr: np.ndarray, torch_arr: np.ndarray) -> None:
    a = jax_arr.flatten().astype(np.float64)
    b = torch_arr.flatten().astype(np.float64)
    diff = np.abs(a - b)
    print(f"\n{name}")
    print(f"  mean abs: {diff.mean():.6f}")
    print(f"  max abs : {diff.max():.6f}")
    for tol in (1e-2, 5e-2, 1e-1):
        within = diff <= (tol + tol * np.abs(a))
        print(f"  within tol={tol:g}: {within.mean()*100:.2f}%")


print(f"Using SEQ_LEN={SEQ_LEN} (bf16 ground truth)")
print(f"JAX_DEVICE={JAX_DEVICE}")
print(f"ALPHAGENOME_TORCH_BF16={os.environ.get('ALPHAGENOME_TORCH_BF16','0')}")

np.random.seed(SEED)
seq_np = np.random.randint(0, 4, (1, SEQ_LEN))

jax_device = pick_jax_device()
print(f"JAX device: {jax_device}")

jax_model = create_from_huggingface(model_version="all_folds", device=jax_device)
metadata = jax_model._metadata
model_settings = dna_model_lib.ModelSettings()

# Ground truth: JAX bf16 compute/output
jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')

@hk.transform_with_state
def forward(dna_sequence, organism_index):
    with hk.mixed_precision.push_policy(model_lib.AlphaGenome, jmp_policy):
        return model_lib.AlphaGenome(
            metadata,
            num_splice_sites=model_settings.num_splice_sites,
            splice_site_threshold=model_settings.splice_site_threshold,
        )(dna_sequence, organism_index)

seq_onehot = jnp.eye(4, dtype=jnp.float32)[seq_np]
organism_jax = jnp.zeros((1,), dtype=jnp.int32)

(preds, embeds), _ = forward.apply(jax_model._params, jax_model._state, None, seq_onehot, organism_jax)

jax_out = {
    'embeds_1bp': np.asarray(embeds.embeddings_1bp),
    'embeds_128bp': np.asarray(embeds.embeddings_128bp),
    'embeds_pair': np.asarray(embeds.embeddings_pair),
}

# Convert and load PyTorch
state_dict = convert_checkpoint(flatten_nested_dict(jax_model._params), flatten_nested_dict(jax_model._state), verbose=False)
model = AlphaGenome()
model.load_state_dict(state_dict, strict=False)
model.eval()
set_update_running_var(model, False)

# Choose device for torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    try:
        model = model.to(device)
        print(f"PyTorch device: {device}")
    except torch.OutOfMemoryError:
        print("PyTorch GPU OOM, falling back to CPU")
        model = model.cpu()
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print("PyTorch device: CPU")

seq_t = torch.from_numpy(seq_np).to(device)
org_t = torch.zeros(1, dtype=torch.long, device=device)

# PyTorch float32
with torch.no_grad():
    embeds_fp32 = model(seq_t, org_t, return_embeds=True)

fp32_out = {
    'embeds_1bp': embeds_fp32.embeds_1bp.detach().cpu().numpy(),
    'embeds_128bp': embeds_fp32.embeds_128bp.detach().cpu().numpy(),
    'embeds_pair': embeds_fp32.embeds_pair.detach().cpu().numpy(),
}

print("\n=== JAX bf16 vs PyTorch float32 ===")
for name in ('embeds_1bp','embeds_128bp','embeds_pair'):
    stats_report(name, jax_out[name], fp32_out[name])

# PyTorch AMP bf16
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds_bf16 = model(seq_t, org_t, return_embeds=True)
    bf16_out = {
        'embeds_1bp': embeds_bf16.embeds_1bp.detach().cpu().numpy(),
        'embeds_128bp': embeds_bf16.embeds_128bp.detach().cpu().numpy(),
        'embeds_pair': embeds_bf16.embeds_pair.detach().cpu().numpy(),
    }
    print("\n=== JAX bf16 vs PyTorch AMP bf16 ===")
    for name in ('embeds_1bp','embeds_128bp','embeds_pair'):
        stats_report(name, jax_out[name], bf16_out[name])
else:
    print("\nPyTorch AMP bf16 not available on this device.")
