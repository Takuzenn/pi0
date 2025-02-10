import torch
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version

if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )

ifPrint = False

# @torch.compile(dynamic=False)
def flex_attention_forward(
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling=None,
):
    """
    This is defined out of classes to make compile happy.
    """

    original_dtype = query_states.dtype
    num_att_heads = 8
    num_key_value_heads = 1
    num_key_value_groups = num_att_heads // num_key_value_heads
    if ifPrint: print(f"key_states shape is {key_states.shape}")
    key_states = key_states[:, :, :, None, :]
    if ifPrint: print(f"key_states shape is {key_states.shape}")
    key_states = key_states.expand(
        batch_size, key_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    if ifPrint: print(f"key_states shape is {key_states.shape}")
    key_states = key_states.reshape(
        batch_size, key_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )
    if ifPrint: print(f"key_states shape is {key_states.shape}")

    value_states = value_states[:, :, :, None, :]
    value_states = value_states.expand(
        batch_size, value_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, value_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    if ifPrint: print(f"key_states shape is {key_states.shape}")

    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    if ifPrint: print(f"attention_mask shape is {attention_mask.shape}")
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]
        if ifPrint: print(f"causal_mask shape is {causal_mask.shape}")
        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)
            if ifPrint: print(f"causal_mask shape is {causal_mask.shape}")

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    b_mask, h_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask
    if ifPrint: print(f"q_len is {q_len}")
    if ifPrint: print(f"kv_len is {kv_len}")
    block_size = 128
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    if ifPrint: print(f"q_len_rounded is {q_len_rounded}")
    if ifPrint: print(f"kv_len_rounded is {kv_len_rounded}")
    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    if ifPrint: print(f"pad_q is {pad_q}")
    pad_k = kv_len_rounded - kv_len
    if ifPrint: print(f"pad_k is {pad_k}")

    if ifPrint: print(f"causal_mask shape is {causal_mask.shape}")
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    if ifPrint: print(f"padded_causal_mask shape is {padded_causal_mask.shape}")

    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)
    if ifPrint: print(f"mask_mod_fn_orig shape is {mask_mod_fn_orig.shape}")

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
        _compile=False,
    )
    if ifPrint: print(f"mask_4d shape is {mask_4d.shape}")

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    if ifPrint: print(f"mask_mod_fn_padded shape is {mask_mod_fn_padded.shape}")

    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )
    if ifPrint: print(f"block_mask shape is {block_mask.shape}")

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output, attention_weights = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,  # because we shaped query/key states for GQA
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
    )
    if ifPrint: print(f"attn_output shape is {attn_output.shape}")
    if ifPrint: print(f"attention_weights shape is {attention_weights.shape}")

    attn_output = attn_output.to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    if ifPrint: print(f"attn_output shape is {attn_output.shape}")

    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    if ifPrint: print(f"attn_output shape is {attn_output.shape}")

    return attn_output
