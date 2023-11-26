import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn_lade import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from flash_attn_lade.bert_padding import pad_input, unpad_input
from flash_attn_lade.flash_attn_interface import _get_block_size
#from flash_attn_lade.layers.rotary import apply_rotary_emb

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    attention_mask=None,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))

    if attention_mask is not None:
        #print("size: ", attention.size(), attention_mask.size())
        scores = scores + attention_mask

    attention = torch.softmax(scores, dim=-1)

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
    )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()



'''
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [128])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (3, 1024),
        (1, 339),
        (64, 800),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (16, 100000),
        (128, 128),
        (256, 256),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_splitkv(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)
    out, lse, _ = flash_attn_func(
        q, k, v, 0.0, causal=causal, window_size=window_size, return_attn_probs=True, lookahead=[0,0,0,0,0, 0, 0]
    )
    out_ref, attn_ref = attention_ref(
        q, k, v, None, None, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    
    #print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    #print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    #print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    #print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
'''


def generate_b(b, window, level, guess, sz, fill_offset, guess_offset):
    for i in range(sz):
        for j in range(sz):
            if j > i:
                b[i][j] = 1 #0 for purple
            else:
                b[i][j] = 0 #1 for yellow

    wl = window * (level - 1) + fill_offset
    for i in range(sz):
        left = guess_offset
        right = (i - (i - wl) % (level - 1))
        if i < wl:
            div = (i - window - fill_offset) // (level - 2)
            left = div + fill_offset
            right = window + div * (level - 2) + fill_offset

        for j in range(sz):
            if j > left and j < right and i >= window + fill_offset:
                b[i][j] = 1
    
    return b 

'''
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [128])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("swap_sq_sk", [False])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (3, 1024),
        (1, 339),
        (64, 800),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (16, 100000),
        (128, 128),
        (256, 256),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_splitkv(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)
    out, lse, _ = flash_attn_func(
        q, k, v, 0.0, causal=causal, window_size=window_size, return_attn_probs=True, lookahead=[0,0,0,0,0,0,0]
    )
        

    out_ref, attn_ref = attention_ref(
        q, k, v, None, None, 0.0, None, causal=causal, window_size=window_size, 
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    
    #print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    #print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    #print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    #print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
'''
@pytest.mark.parametrize("fill_offset", [0,1,4,5,8])
@pytest.mark.parametrize("guess_offset", [0,1,2,3,4])
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [128])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64])
#@pytest.mark.parametrize("window", [1,2,3,4,5,6,7,10,16,32,64,66,79,128,155])
#@pytest.mark.parametrize("guess", [1,2,3,4,5,6,7,10,16,32,64,66,79,128,155])
#@pytest.mark.parametrize("kv_cache", [0,1,2,3,4,5,6,7,10,16,32,64,66,79,128,155,240,320,340,546,1024])
#@pytest.mark.parametrize("level", [3,4,5,6,7,8,9,10,11,12,13,14])
#@pytest.mark.parametrize("fill_offset", [0,1,2,3,4,5,6,7,8])
#@pytest.mark.parametrize("guess_offset", [0,1,2,3,4,5,6,7,8])
@pytest.mark.parametrize("window", [1,2,3,5,6,66,155])
@pytest.mark.parametrize("guess", [1,2,3,4,66,79,128,155])
@pytest.mark.parametrize("kv_cache", [0,1,2,3,16,32,155,340,546,1024])
@pytest.mark.parametrize("level", [3,4,7,10,14])
# @pytest.mark.parametrize("swap_sq_sk", [False])
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_splitkv_lookahead(d, causal, local, dtype, window, level, guess, kv_cache, fill_offset, guess_offset):
    seqlen_q = window * (level - 1) + (level - 1) * guess + fill_offset
    seqlen_k = kv_cache + seqlen_q
    real_guess_offset = min(guess_offset, fill_offset)
    #2 3 1 0 1 0
    print("setting: ",  f"{window} {level} {guess} {kv_cache} {fill_offset} {real_guess_offset}")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.rand(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.rand(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.rand(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=False)

    out, lse, _ = flash_attn_func(
        q, k, v, 0.0, causal=causal, window_size=window_size, return_attn_probs=True, lookahead=[window, level, guess, kv_cache, fill_offset, real_guess_offset, 0]
    )
    #print(out[0,:,0,0])
    mask = torch.zeros(1, 1, seqlen_q, seqlen_k)
    generate_b(mask[0][0][-seqlen_q:,-seqlen_q:], window, level, guess, seqlen_q, fill_offset, real_guess_offset)
    mask1 = mask.clone().detach()
    attention_mask = mask1.to(torch.float).masked_fill(mask.to(torch.bool), torch.finfo(torch.float).min).to(device)
    #print("mask: ", attention_mask)
    out_ref, attn_ref = attention_ref(
        q, k, v, None, None, 0.0, None, causal=causal, window_size=window_size, attention_mask=attention_mask
    )
    #print(out_ref[0,:,0,0], (attention_mask == 0) * 1)
    attention_mask = mask.to(dtype).masked_fill(mask.to(torch.bool), torch.finfo(dtype).min).to(device)

    attention_mask = attention_mask
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        attention_mask=attention_mask,
        upcast=False,
        reorder_ops=True,
    )
    #print("max: ", out.max(), out_ref.max(), out_pt.max())
    
    #print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    #print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    #print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    #print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5, f"{window} {level} {guess} {kv_cache} {fill_offset} {guess_offset}"


