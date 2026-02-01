# Debug Notes: JAX -> PyTorch Conversion (Merged)

This document merges prior debug notes with updates from 2026-01-31.

## Status (as of 2026-01-31)
- Level2 integration tests pass under JAX float32 policy (default in tests).
- There are no Level3/Level4 tests in `tests/test_integration_jax_torch.py`.
- JAX bfloat16 mixed precision comparisons still diverge (especially in pairwise path).
- After FlashAttention2 + env work, 4KB bfloat16 integration tests mostly pass (see 2026-01-31 update below), but a few strict statistical tests still fail.

## Key Findings (current)
- PyTorch matches JAX float32 outputs essentially exactly.
- Divergence appears when JAX uses mixed precision bfloat16.
- JAX bfloat16 vs PyTorch bfloat16 diverges most in the pairwise path, likely due to kernel/precision differences.

## Diagnostics
### diagnose_precision_mismatch.py (seq_len=4096)
- Float32: embeds_1bp ~99.48-99.56%, embeds_128bp ~98.98-98.99%, embeds_pair ~98.44-98.63%.
- BFloat16: consistently worse than float32.

### Level2 (historical)
- Earlier: embeds_1bp ~98.18%, embeds_128bp ~98.50%, embeds_pair ~97.94% (target >= 99%).
- 2026-01-31: embeds_1bp 98.34%, embeds_128bp 98.56%, embeds_pair 97.81% (before float32 policy change).

## Changes Applied (current code state)
- Fixed `relative_shift` off-by-one in `alphagenome_pytorch/alphagenome.py`.
- Attention scaling matches JAX order (scale after dot).
- Disabled TF32 for CUDA matmul/cudnn to avoid reduced precision paths.
- BatchRMSNorm uses `running_var + eps` (matches JAX).
- Added optional bf16 emulation hooks (disabled by default):
  - enable with `ALPHAGENOME_EMULATE_BF16=1`
- Head mapping fix:
  - genome track heads live under `heads.{organism}.{head}.resolutions.resolution_{res}`
  - updated in `alphagenome_pytorch/convert/mapping_spec.py`

## Test Updates
- Track prediction parity tests added:
  - genome tracks: rna_seq, cage, dnase, procap, atac, chip_tf, chip_histone
  - splice heads: classification logits, usage predictions
  - contact_maps predictions
- Run with:
  - `ALPHAGENOME_RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_integration_jax_torch.py::TestTrackPredictionsLevel2 -v --tb=short`
 - Note: needs Hugging Face auth; run with `set -a && source .env && set +a` to load token.
 - GPU selection override: `ALPHAGENOME_GPU_INDEX=...` (useful if a GPU yields NaNs).
 - Optional pinning example: `CUDA_VISIBLE_DEVICES=3 ALPHAGENOME_GPU_INDEX=0` (stable in this env).
 - Track tests now include NaN/Inf checks per output.

## Environment / Policy Notes
- Tests now default to JAX float32 for deterministic parity.
- Set `ALPHAGENOME_JAX_COMPUTE_DTYPE=bfloat16` to re-enable bfloat16 comparisons.
- Local HF model path must point to a *snapshot* directory (not the repo root), e.g.:
  `/local1/tuxm/huggingface/hub/models--google--alphagenome-all-folds/snapshots/<hash>`
- JAX GPU runs require CuDNN >= 9.8 when using jaxlib 0.9.0; we upgraded to `nvidia-cudnn-cu12==9.18.1.3` in the uv venv.

## Historical Investigations (resolved/clarified)
### Q head layout
- Earlier suspicion: Q head split/reshape mismatch.
- Update: Q projection/head layout matches JAX; not the culprit.

### Transformer vs Encoder (earlier)
- Pre-transformer trunk was close but not exact (~98.95% at seq_len=4096).
- Post-transformer trunk dropped further (~94.28%).
- Pairwise activations were closer (~98.44%).
- These gaps are now explained by mixed-precision (bfloat16) differences.

## Commands Run (latest)
- `ALPHAGENOME_RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_integration_jax_torch.py::TestLevel2Loose -v --tb=short`
- `uv run --extra convert python scripts/diagnose_precision_mismatch.py`
- `SEQ_LEN=16384 ALPHAGENOME_TORCH_BF16=1 uv run --extra convert python scripts/compare_precision_modes.py`
- (2026-01-31) `ALPHAGENOME_JAX_COMPUTE_DTYPE=bfloat16 ALPHAGENOME_GPU_INDEX=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ALPHAGENOME_RUN_INTEGRATION_TESTS=1 .venv/bin/pytest tests/test_integration_jax_torch.py -v -s`
- (2026-01-31) FlashAttention2 benchmark (seq_len=4096): `ALPHAGENOME_TORCH_BF16=1 .venv/bin/python scripts/bench_flash_vs_baseline.py` (see notes below)

## Next Steps
1. If bfloat16 parity is required, implement stricter JAX-style bf16 emulation in pairwise blocks (round after each op).
2. Otherwise, keep tests in float32 mode for deterministic parity.

---

## 2026-01-31 Update (FlashAttention2 + env fixes)
### Environment fixes
- Installed FlashAttention2: `flash-attn==2.8.3` with `torch==2.5.1+cu121`.
- Upgraded cuDNN in venv to resolve JAX init errors:
  `nvidia-cudnn-cu12==9.18.1.3` (jaxlib 0.9.0 expects >= 9.8).
- Important: set `ALPHAGENOME_MODEL_PATH` to the snapshot directory, *not* repo root:
  `/local1/tuxm/huggingface/hub/models--google--alphagenome-all-folds/snapshots/a8f293a76ee73d5b57f3bf2ae146510589fcf187`

### GPU cleanup
- Cleared all GPU processes before running tests (previous Jupyter kernel was holding >15GB).

### Integration tests @ 4KB (bf16)
Command used:
`ALPHAGENOME_JAX_COMPUTE_DTYPE=bfloat16 ALPHAGENOME_GPU_INDEX=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ALPHAGENOME_RUN_INTEGRATION_TESTS=1 .venv/bin/pytest tests/test_integration_jax_torch.py -v -s`

Results:
- **58 passed, 3 failed**
  - `TestForwardPass::test_jax_torch_output_comparison` (max diff ~0.159, mean diff ~0.0016)
  - `TestStatisticalCriteria::test_correlation_embeds_128bp` (corr ~0.999847 < 0.9999)
  - `TestStatisticalCriteria::test_no_systematic_bias` (embeds_1bp bias p < 0.01)
- All Level2Loose checks and track head parity tests **passed** at 4KB/bf16.

### FlashAttention2 benchmark (seq_len=4096)
Setup: full AlphaGenome, batch=1, `return_embeds=True`, 5 warmup + 20 runs.
- Flash path active (pairwise row attention only).
- Baseline mean ~46.76 ms, Flash mean ~47.05 ms → ~0.99× (no speedup at 4KB).
  Likely limited because the main attention path still falls back due to bias/softcap.

### Reproduce (local)
Integration tests (4KB bfloat16, after freeing GPUs):
```
ALPHAGENOME_MODEL_PATH=/local1/tuxm/huggingface/hub/models--google--alphagenome-all-folds/snapshots/a8f293a76ee73d5b57f3bf2ae146510589fcf187 \
ALPHAGENOME_JAX_COMPUTE_DTYPE=bfloat16 \
ALPHAGENOME_GPU_INDEX=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
ALPHAGENOME_RUN_INTEGRATION_TESTS=1 \
.venv/bin/pytest tests/test_integration_jax_torch.py -v -s
```

FlashAttention2 benchmark (script + multi-length run):
```
ALPHAGENOME_TORCH_BF16=1 .venv/bin/python scripts/bench_flash_vs_baseline.py \
  --seq-lens 4096 8192 16384 --batch 1 --runs 20 --warmup 5
```

### Benchmark results (2026-01-31, bf16)
- seq_len=4096: baseline 46.44 ms, flash 47.18 ms → 0.984×
- seq_len=8192: baseline 70.47 ms, flash 70.51 ms → 0.999×
- seq_len=16384: baseline 127.85 ms, flash 127.99 ms → 0.999×

### FlexAttention constraints (torch 2.5)
- `torch.nn.attention.flex_attention` currently requires power-of-two head dims and `dim_head_v <= dim_head_qk`.
- Default AlphaGenome uses `dim_head_qk=128` and `dim_head_v=192`, so FlexAttention cannot activate (falls back).

### Track prediction parity (4KB, bf16)
Command used:
`ALPHAGENOME_JAX_COMPUTE_DTYPE=bfloat16 ALPHAGENOME_GPU_INDEX=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ALPHAGENOME_RUN_INTEGRATION_TESTS=1 .venv/bin/pytest tests/test_integration_jax_torch.py::TestTrackPredictionsLevel2 -v --tb=short`

Results:
- **16 passed** (all track outputs within tolerance)

### Benchmarks (longer seq, bf16)
Command used:
`ALPHAGENOME_TORCH_BF16=1 .venv/bin/python scripts/bench_flash_vs_baseline.py --seq-lens 4096 16384 65536 131072 196608 262144 --batch 1 --runs 5 --warmup 2 --measure-mem`

Results (mean ms, speedup = baseline / flash):
- seq_len=4096: baseline 46.98 ms, flash 47.53 ms → 0.988×; peak alloc 1.75 GB vs 3.27 GB
- seq_len=16384: baseline 126.86 ms, flash 127.51 ms → 0.995×; peak alloc 3.56 GB (both)
- seq_len=65536: baseline 461.34 ms, flash 463.72 ms → 0.995×; peak alloc 5.12 GB (both)
- seq_len=131072: baseline 917.34 ms, flash 922.61 ms → 0.994×; peak alloc 7.19 GB (both)
- seq_len=196608: baseline 1399.73 ms, flash 1394.97 ms → 1.003×; peak alloc 9.27 GB (both)
- seq_len=262144: baseline 1880.99 ms, flash 1892.89 ms → 0.994×; peak alloc 11.34 GB (both)

### Baseline-only large seq (bf16)
Command used:
`ALPHAGENOME_TORCH_BF16=1 .venv/bin/python - <<'PY' ...`

Results:
- seq_len=524288 (512KB): baseline OK, ~4450.8 ms, peak alloc 18.13 GB, reserved 18.96 GB
- seq_len=1048576 (1MB): baseline OOM

### torch.compile (baseline, bf16)
Command used:
`ALPHAGENOME_TORCH_BF16=1 .venv/bin/python - <<'PY' ...`

Result:
- seq_len=131072: baseline ~923.5 ms, compiled ~808.2 ms → **1.14×** speedup
- Noted torch._dynamo cache-size warnings and TF32 advisory (kept TF32 disabled for parity).

### FlexAttention benchmarks (bf16)
Default model (flex path enabled via padding to next power-of-two dims, low-res bias indexed in score_mod):
`ALPHAGENOME_TORCH_BF16=1 .venv/bin/python scripts/bench_flex_vs_baseline.py --seq-lens 4096 16384 65536 --batch 1 --runs 3 --warmup 1 --measure-mem`

Results:
- Flex used? **Yes** (main attention path)
- seq_len=4096: baseline 48.31 ms, flex 95.91 ms → 0.50× (slower)
- seq_len=16384: baseline 125.38 ms, flex 176.77 ms → 0.71× (slower)
- seq_len=65536: baseline 458.15 ms, flex 497.08 ms → 0.92× (slower)

Flex-compatible variant (dim_head_qk=128, dim_head_v=128; no padding needed):
`ALPHAGENOME_TORCH_BF16=1 .venv/bin/python scripts/bench_flex_vs_baseline.py --flex-compatible --seq-lens 4096 16384 65536 --batch 1 --runs 5 --warmup 2 --measure-mem`

Results:
- Flex used? **Yes** (main attention path)
- seq_len=4096: baseline 47.15 ms, flex 73.62 ms → 0.64× (slower)
- seq_len=16384: baseline 127.54 ms, flex 154.85 ms → 0.82× (slower)
- seq_len=65536: baseline 464.00 ms, flex 483.85 ms → 0.96× (slightly slower)

### torch.compile + FlexAttention
Nightly venv (torch 2.11.0.dev20260131+cu128, triton 3.6.0):
Command used:
`ALPHAGENOME_TORCH_BF16=1 .venv-nightly/bin/python - <<'PY' ...`

Results:
- seq_len=4096: baseline compiled **36.95 ms**; flex compiled **failed** (shared memory OOR: required 163840 > 101376)
- seq_len=16384: baseline compiled **112.63 ms**; flex compiled **165.66 ms** → **0.77×** vs baseline eager (still slower)
- seq_len=131072: baseline compiled **270.61 ms**; flex compiled **failed** (shared memory OOR: required 167936 > 101376)
- seq_len=196608: baseline compiled **450.35 ms**; flex compiled **495.45 ms** (slower than baseline compiled)
- seq_len=262144: baseline compiled **613.90 ms**; flex compiled **680.88 ms** (slower than baseline compiled)
