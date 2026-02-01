# Analysis Notes: Performance Breakdown

## Encoder vs Transformer vs Decoder (TF32 + BF16)
These measurements split the AlphaGenome UNet into:
- **Encoder**: DNAEmbed + downsampling blocks
- **Transformer**: 1D/2D attention tower (on bp/128 tokens)
- **Decoder**: upsampling blocks back to 1bp resolution

Command (tf32 + bf16):
`ALPHAGENOME_ALLOW_TF32=1 ALPHAGENOME_TORCH_BF16=1 .venv/bin/python - <<'PY' ...`

### seq_len=4096 (bp input, 32 transformer tokens)
- encoder (embed+downs): **7.51 ms**
- transformer only: **18.13 ms** (25.64 − 7.51)
- decoder only: **6.43 ms** (32.07 − 25.64)
- full unet: **31.99 ms**

**Bottleneck:** transformer

### seq_len=262144 (bp input, 2048 transformer tokens)
- encoder (embed+downs): **285.64 ms**
- transformer only: **78.65 ms** (364.29 − 285.64)
- decoder only: **364.75 ms** (729.04 − 364.29)
- full unet: **731.40 ms**

**Bottleneck:** decoder + encoder (conv paths at base‑pair resolution)
