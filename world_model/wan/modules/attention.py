# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def _build_sdpa_mask(q_lens, k_lens, Lq, Lk, device, window_size=(-1,-1), causal=False):
    """
    Returns attn_mask of shape [B, 1, Lq, Lk] where True means *DISALLOWED* (PyTorch SDPA convention).
    """
    B = q_lens.numel()
    q_idx = torch.arange(Lq, device=device)[None, :]          # [1, Lq]
    k_idx = torch.arange(Lk, device=device)[None, :]          # [1, Lk]
    q_valid = (q_idx < q_lens.view(B,1))                      # [B, Lq]
    k_valid = (k_idx < k_lens.view(B,1))                      # [B, Lk]
    valid = q_valid[:, :, None] & k_valid[:, None, :]         # [B, Lq, Lk]

    if causal:
        valid &= (k_idx <= q_idx[:, :, None])

    left, right = window_size
    if (left, right) != (-1, -1):
        valid &= (k_idx >= (q_idx[:, :, None] - left))
        valid &= (k_idx <= (q_idx[:, :, None] + right))

    attn_mask = ~valid                                          # True = mask OUT
    return attn_mask.unsqueeze(1)                               # [B, 1, Lq, Lk]


def _jyx_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1,-1),
    deterministic=False, dtype=torch.bfloat16, fa_version=None,
):
    """
    q,k,v: [B, L, H, D]  (same as your WanSelfAttention uses after qkv_fn)
    Returns: [B, L, H, D]
    """
    use_fa = (FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE)
    if use_fa:
        return flash_attention(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size,
            dtype=dtype, version=fa_version,
        )

    # ---- SDPA fallback with correct varlen/window/causal ----
    if q_lens is None or k_lens is None:
        warnings.warn("No q_lens/k_lens provided; SDPA will attend to padding.")
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    device = q.device

    # Build mask: True = disallowed (PyTorch convention)
    if q_lens is not None and k_lens is not None:
        attn_mask = _build_sdpa_mask(
            q_lens.to(device), k_lens.to(device),
            Lq=Lq, Lk=Lk, device=device,
            window_size=window_size, causal=causal
        )  # [B,1,Lq,Lk]
    else:
        attn_mask = None

    # SDPA expects [B,H,L,D]
    q_ = q.transpose(1,2).to(dtype)
    k_ = k.transpose(1,2).to(dtype)
    v_ = v.transpose(1,2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_mask, is_causal=False, dropout_p=dropout_p
    )  # [B,H,Lq,D]

    return out.transpose(1,2).contiguous()