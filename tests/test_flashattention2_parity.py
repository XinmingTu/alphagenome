from __future__ import annotations

import importlib
import os
import importlib.util

import pytest
import torch

os.environ.setdefault("ALPHAGENOME_TORCH_BF16", "1")

_FLASH_AVAILABLE = importlib.util.find_spec("flash_attn") is not None

pytestmark = pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_RUN_FLASH_ATTN_TESTS", "0") != "1",
    reason="FlashAttention2 tests disabled. Set ALPHAGENOME_RUN_FLASH_ATTN_TESTS=1 to enable."
)


def _reload_alphagenome():
    import alphagenome_pytorch.alphagenome as ag
    return importlib.reload(ag)


def _flash_used(model) -> bool:
    for layer in model.transformer_unet.transformer.layers:
        attn = layer[0].block
        if getattr(attn, "last_attn_used_flash", False):
            return True
        pairwise_wrapper = layer[3]
        if pairwise_wrapper is not None:
            pairwise_attn = pairwise_wrapper.block
            if getattr(pairwise_attn, "last_attn_used_flash", False):
                return True
    return False


def _assert_nested_close(a, b):
    if a is None or b is None:
        assert a is None and b is None
        return
    if isinstance(a, torch.Tensor):
        use_bf16 = os.environ.get("ALPHAGENOME_TORCH_BF16", "0") == "1"
        if a.dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 5e-2, 5e-2
        elif use_bf16:
            atol, rtol = 2e-3, 2e-3
        else:
            atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a.keys():
            _assert_nested_close(a[key], b[key])
        return
    raise AssertionError(f"Unsupported output type: {type(a)}")


def test_flashattention2_matches_baseline():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FlashAttention2 parity test.")
    if not _FLASH_AVAILABLE:
        pytest.skip("flash-attn not installed.")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this GPU; flash-attn path will not activate.")

    ag = _reload_alphagenome()

    torch.manual_seed(0)

    transformer_kwargs = {
        "depth": 2,
        "heads": 4,
        "dim_head_qk": 32,
        "dim_head_v": 32,
        "dropout": 0.0,
        "ff_expansion_factor": 2.0,
        "max_positions": 256,
        "dim_pairwise": 32,
        "pairwise_every_num_single_blocks": 1,
        "single_to_pairwise_heads": 4,
        "pool_size": 4,
    }

    genome_track_heads = {
        "rna_seq": dict(num_tracks=2, resolutions=(1, 128)),
        "cage": dict(num_tracks=2, resolutions=(1, 128)),
        "dnase": dict(num_tracks=2, resolutions=(1, 128)),
        "procap": dict(num_tracks=2, resolutions=(1, 128)),
        "atac": dict(num_tracks=2, resolutions=(1, 128)),
        "chip_tf": dict(num_tracks=2, resolutions=(128,)),
        "chip_histone": dict(num_tracks=2, resolutions=(128,)),
    }

    base_model = ag.AlphaGenome(
        dims=(64, 128),
        basepairs=4,
        dna_embed_width=7,
        num_organisms=1,
        transformer_kwargs={**transformer_kwargs, "use_flash_attn": False},
    )
    base_model.add_reference_heads(
        "human",
        genome_track_heads=genome_track_heads,
        num_tracks_contacts=2,
        num_splice_usage_tracks=4,
        splice_junction_hidden_dim=8,
        splice_junction_max_tissues=4,
    )

    flash_model = ag.AlphaGenome(
        dims=(64, 128),
        basepairs=4,
        dna_embed_width=7,
        num_organisms=1,
        transformer_kwargs={**transformer_kwargs, "use_flash_attn": True},
    )
    flash_model.add_reference_heads(
        "human",
        genome_track_heads=genome_track_heads,
        num_tracks_contacts=2,
        num_splice_usage_tracks=4,
        splice_junction_hidden_dim=8,
        splice_junction_max_tissues=4,
    )
    flash_model.load_state_dict(base_model.state_dict())

    device = torch.device("cuda")
    base_model = base_model.to(device).eval()
    flash_model = flash_model.to(device).eval()

    batch = 2
    seq_len = 128
    dna = torch.randint(0, 4, (batch, seq_len), device=device)

    with torch.no_grad():
        out_base = base_model(dna, organism_index=0)
        out_flash = flash_model(dna, organism_index=0)

    if not _flash_used(flash_model):
        pytest.skip("FlashAttention2 path not active (no compatible kernel used).")

    _assert_nested_close(out_base, out_flash)
