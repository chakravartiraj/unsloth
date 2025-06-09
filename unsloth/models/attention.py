import torch
import torch.nn as nn
from typing import Optional, Tuple

from unsloth.kernels.utils import HAS_FLASH_ATTENTION, HAS_XFORMERS

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

# It seems xformers_attention is a wrapper/utility in unsloth.kernels.utils
# If direct xformers.ops.memory_efficient_attention is needed, that can be imported too.
# For now, following the convention observed in other unsloth files.
_xformers_memory_efficient_attention_forward_only = None
if HAS_XFORMERS:
    # Standard xformers import
    import xformers.ops as xops
    from unsloth.kernels.utils import xformers_attention # This is likely xops.memory_efficient_attention

    try:
        from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    except ImportError:
        BlockDiagonalCausalMask = None
        pass

    try:
        # Attempt to import the specific forward-only variant if available
        from xformers.ops.fmha import memory_efficient_attention_forward as _xformers_memory_efficient_attention_forward_only
    except ImportError:
        pass # It's okay if this specific variant isn't available

from unsloth.kernels import inplace_rope_embedding
try:
    from unsloth.models.llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, \
                                    LlamaYaRNScalingRotaryEmbedding, LongRopeRotaryEmbedding
except:
    # Might not exist for all versions, provide stubs if so.
    # This is unlikely to be an issue with a fixed environment.
    class LlamaRotaryEmbedding: pass
    class LlamaLinearScalingRotaryEmbedding: pass
    class LlamaYaRNScalingRotaryEmbedding: pass
    class LongRopeRotaryEmbedding: pass

from torch.nn.functional import scaled_dot_product_attention


class BaseAttention(nn.Module):
    def __init__(self, config, model_name: str = "model"):
        super().__init__()
        self.config = config
        self.model_name = model_name # To distinguish between different model's attention mechanisms if needed

        # Initialize common parameters - these might be derived from config
        # For example, in Llama:
        # self.hidden_size = config.hidden_size
        # self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        # self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size: int = getattr(config, "hidden_size", 0)
        self.num_attention_heads: int = getattr(config, "num_attention_heads", 0)
        self.num_key_value_heads: int = getattr(config, "num_key_value_heads", self.num_attention_heads)

        if self.num_attention_heads > 0:
            self.head_dim: int = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        elif self.hidden_size > 0 and self.num_key_value_heads > 0: # Support for MQA/GQA where num_attention_heads might not be the primary factor for head_dim
            self.head_dim: int = getattr(config, "head_dim", self.hidden_size // self.num_key_value_heads) # This might be wrong, usually head_dim is tied to num_attention_heads
        else:
            self.head_dim: int = getattr(config, "head_dim", 0)

        if self.hidden_size == 0 or self.num_attention_heads == 0 or self.head_dim == 0 :
             # Subclasses must ensure these are properly initialized if not available in config.
             # For Llama, these are typically always available.
            pass

        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads if self.num_key_value_heads > 0 and self.num_attention_heads > 0 else 1


    def _init_kv_cache(self, batch_size: int, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        """
        Initializes Key-Value (KV) cache.
        This method should be implemented or overridden by subclasses.
        """
        # Example:
        # self.k_cache = torch.zeros((batch_size, max_seq_len, self.num_key_value_heads, self.head_dim), dtype=dtype, device=device)
        # self.v_cache = torch.zeros((batch_size, max_seq_len, self.num_key_value_heads, self.head_dim), dtype=dtype, device=device)
        raise NotImplementedError("KV caching must be implemented in subclasses.")

    def _update_kv_cache(self, key_states: torch.Tensor, value_states: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]], position_ids: Optional[torch.LongTensor] = None):
        """
        Updates and returns the KV cache.
        This method should be implemented or overridden by subclasses.
        """
        # Example:
        # if past_key_value is None or position_ids is None:
        #     # Handle initialization or no caching scenario
        #     return key_states, value_states
        #
        # current_seq_len = key_states.shape[1]
        # self.k_cache = self.k_cache.to(key_states.device)
        # self.v_cache = self.v_cache.to(value_states.device)
        #
        # self.k_cache[:key_states.shape[0], position_ids, :, :] = key_states
        # self.v_cache[:value_states.shape[0], position_ids, :, :] = value_states
        #
        # cached_k = self.k_cache[:key_states.shape[0], :position_ids + current_seq_len]
        # cached_v = self.v_cache[:value_states.shape[0], :position_ids + current_seq_len]
        # return cached_k, cached_v
        raise NotImplementedError("KV caching must be implemented in subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        is_inference_pass: bool = False, # New parameter for memory-efficient forward
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention.
        Supports both padded BxSxH inputs and packed (unpadded) TxH inputs.
        Can also accept pre-computed and packed QKV tensor.

        Parameters:
            hidden_states (torch.Tensor): Input tensor. Shape (batch_size, seq_len, hidden_size) for padded,
                                          or (total_tokens, hidden_size) for packed. Ignored if packed_qkv is provided.
            attention_mask (Optional[torch.Tensor]): Mask for padded attention. Not typically used for packed.
            position_ids (Optional[torch.LongTensor]): Position IDs for RoPE. Crucial for packed.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): KV cache.
            output_attentions (bool): Whether to output attention weights.
            use_cache (bool): Whether to use KV caching. Disabled for packed input or pre-packed QKV.
            cu_seqlens_q (Optional[torch.Tensor]): Cumulative sequence lengths for query (packed input).
            cu_seqlens_kv (Optional[torch.Tensor]): Cumulative sequence lengths for key/value (packed input).
            max_seqlen_q (Optional[int]): Maximum query sequence length in the batch (packed input).
            max_seqlen_kv (Optional[int]): Maximum key/value sequence length in the batch (packed input).
            packed_qkv (Optional[torch.Tensor]): Pre-computed QKV tensor. If provided, q_proj, k_proj, v_proj are skipped.
                                                Expected shape for packed: (total_tokens, num_q_heads + 2*num_kv_heads, head_dim)
                                                Expected shape for padded: (bsz, q_len, num_q_heads + 2*num_kv_heads, head_dim)
            is_inference_pass (bool): If True, signals that gradients are not required, allowing for memory-efficient xformers ops.
        This method should be implemented by subclasses using one of the attention mechanisms.
        """
        # Placeholder for the forward pass logic.
        # Subclasses will implement the specific attention calculation (Flash, Xformers, SDPA).

        # 1. Project hidden_states to Q, K, V
        # query_states = self.q_proj(hidden_states)
        # key_states   = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)

        # 2. Reshape Q, K, V for multi-head attention
        # query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        # key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. KV Caching
        # if use_cache:
        #    if past_key_value is not None:
        #        # Reuse k, v, self_attention
        #        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #        value_states = torch.cat([past_key_value[1], value_states], dim=2)
        #    past_key_value = (key_states, value_states)

        # 4. Repeat K, V if num_key_value_groups > 1 (for GQA/MQA)
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 5. Attention Calculation
        # attn_output = None
        # attn_weights = None
        # if HAS_FLASH_ATTENTION and use_flash_attention:
        #     attn_output = flash_attn_func(query_states, key_states, value_states, attention_mask, ...)
        # elif HAS_XFORMERS and use_xformers:
        #     attn_output = xformers_attention(query_states, key_states, value_states, attention_mask, ...)
        # else: # Default to SDPA
        #     attn_output = scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, ...)

        # 6. Reshape output and project
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # attn_output = self.o_proj(attn_output)

        # if output_attentions:
        #     # Return attn_weights as well, if computed
        #     pass

        # return attn_output, attn_weights, past_key_value
        raise NotImplementedError("Forward pass must be implemented in subclasses.")

# Helper function for GQA/MQA, often found in attention implementations
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Helper to generate position_ids from cu_seqlens
def _get_position_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Generates position_ids for packed sequences based on cu_seqlens.
    Example: cu_seqlens = [0, 3, 5] -> two sequences of len 3 and 2
    Output: [0, 1, 2, 0, 1]
    """
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    position_ids = torch.cat([torch.arange(0, s, dtype=torch.long, device=device) for s in seqlens])
    return position_ids

# Helper to generate SDPA mask for packed sequences
def _create_block_diag_causal_sdpa_mask(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q_total_tokens: int,
    kv_total_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Creates a block-diagonal causal attention mask for SDPA with packed sequences.
    Also incorporates sliding window attention if specified.
    """
    mask = torch.full((1, q_total_tokens, kv_total_tokens), torch.finfo(dtype).min, dtype=dtype, device=device)

    q_start_indices = cu_seqlens_q[:-1]
    q_end_indices = cu_seqlens_q[1:]
    kv_start_indices = cu_seqlens_kv[:-1]
    kv_end_indices = cu_seqlens_kv[1:]

    num_sequences = len(q_start_indices)
    for i in range(num_sequences):
        q_start, q_end = q_start_indices[i], q_end_indices[i]
        kv_start, kv_end = kv_start_indices[i], kv_end_indices[i]

        # Create a causal mask for the current block
        # q_indices are rows, kv_indices are columns in the sub-block
        q_block_len = q_end - q_start
        kv_block_len = kv_end - kv_start

        # Absolute indices for the current block
        q_indices_abs = torch.arange(q_start, q_end, device=device)
        kv_indices_abs = torch.arange(kv_start, kv_end, device=device)

        # Relative indices for causal and sliding window logic
        q_indices_rel = torch.arange(q_block_len, device=device)
        kv_indices_rel = torch.arange(kv_block_len, device=device)

        # Mask setup for the current block
        block_mask = torch.ones((q_block_len, kv_block_len), dtype=torch.bool, device=device)

        # Apply causality: q_rel >= kv_rel
        block_mask = block_mask & (q_indices_rel.unsqueeze(1) >= kv_indices_rel.unsqueeze(0))

        if sliding_window is not None and sliding_window > 0:
            # Apply sliding window: q_rel - kv_rel < sliding_window
            # or kv_rel >= q_rel - sliding_window + 1
            block_mask = block_mask & (kv_indices_rel.unsqueeze(0) >= q_indices_rel.unsqueeze(1) - sliding_window + 1)

        mask[0, q_indices_abs.unsqueeze(1), kv_indices_abs.unsqueeze(0)] = \
            torch.where(block_mask, 0.0, torch.finfo(dtype).min)

    return mask.expand(q_total_tokens, q_total_tokens, kv_total_tokens) # Expected by SDPA (B, M, N) or (M,N) broadcast


class LlamaAttention(BaseAttention):
    def __init__(self, config, **kwargs): # Added **kwargs to pass to BaseAttention if any
        super().__init__(config, model_name="llama", **kwargs)

        # self.hidden_size is already set in BaseAttention if config.hidden_size exists
        # self.num_attention_heads, self.num_key_value_heads, self.head_dim are initialized in BaseAttention
        # self.num_key_value_groups is also initialized in BaseAttention

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=getattr(config, "attention_bias", False))

        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "yarn":
                 self.rotary_emb = LlamaYaRNScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    scaling_factor=scaling_factor,
                    original_max_position_embeddings=self.rope_scaling.get("original_max_position_embeddings", self.max_position_embeddings), # Fallback for yarn
                    beta_fast = self.rope_scaling.get("beta_fast", 32),
                    beta_slow = self.rope_scaling.get("beta_slow", 1),
                )
            elif scaling_type == "longrope":
                self.rotary_emb = LongRopeRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        is_inference_pass: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        is_packed_input = cu_seqlens_q is not None # True if sequence packing (unpadded) is used

        if packed_qkv is not None and use_cache:
            # Pre-packed QKV usually implies inference optimization where KV cache might be handled differently or disabled.
            raise NotImplementedError("KV caching is not supported when `packed_qkv` is provided.")
        if packed_qkv is not None and past_key_value is not None:
             raise ValueError("`past_key_value` cannot be used when `packed_qkv` is provided.")


        if is_packed_input:
            if use_cache: # Redundant check if packed_qkv is also None, but good for clarity
                raise NotImplementedError("KV caching is not supported for packed sequences.")
            if past_key_value is not None: # Same
                raise ValueError("`past_key_value` cannot be used with packed sequences.")

            total_tokens_q = hidden_states.shape[0] if packed_qkv is None else packed_qkv.shape[0]
            bsz, q_len = None, None
        else:
            # Padded inputs
            if packed_qkv is not None:
                bsz, q_len = packed_qkv.shape[0], packed_qkv.shape[1]
            elif hidden_states is not None:
                bsz, q_len, _ = hidden_states.shape
            else:
                raise ValueError("Either hidden_states or packed_qkv must be provided for padded input.")
            total_tokens_q = bsz * q_len

        if packed_qkv is not None:
            # Split pre-packed QKV
            q_slice_end = self.num_attention_heads
            k_slice_end = self.num_attention_heads + self.num_key_value_heads

            if is_packed_input: # 3D packed_qkv (total_tokens, num_total_heads, head_dim)
                query_states = packed_qkv[:, :q_slice_end, :]
                key_states   = packed_qkv[:, q_slice_end:k_slice_end, :]
                value_states = packed_qkv[:, k_slice_end:, :]
            else: # 4D padded_qkv (bsz, q_len, num_total_heads, head_dim)
                query_states = packed_qkv[:, :, :q_slice_end, :]
                key_states   = packed_qkv[:, :, q_slice_end:k_slice_end, :]
                value_states = packed_qkv[:, :, k_slice_end:, :]
                # Transpose to (bsz, num_heads, q_len, head_dim)
                query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()
                key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

            if cu_seqlens_kv is None and is_packed_input: cu_seqlens_kv = cu_seqlens_q
            if max_seqlen_kv is None and is_packed_input: max_seqlen_kv = max_seqlen_q
            total_tokens_kv = total_tokens_q if is_packed_input else None # Handled by key_states.shape later for padded

        else: # Standard path: project from hidden_states
            if hidden_states is None:
                raise ValueError("`hidden_states` must be provided if `packed_qkv` is not.")
            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if is_packed_input:
                query_states = query_states.view(total_tokens_q, self.num_attention_heads, self.head_dim)
                key_states   = key_states.view(total_tokens_q, self.num_key_value_heads, self.head_dim)
                value_states = value_states.view(total_tokens_q, self.num_key_value_heads, self.head_dim)
                if cu_seqlens_kv is None: cu_seqlens_kv = cu_seqlens_q
                if max_seqlen_kv is None: max_seqlen_kv = max_seqlen_q
                total_tokens_kv = total_tokens_q
            else:
                query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
                key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None and is_packed:
            position_ids = _get_position_ids_from_cu_seqlens(cu_seqlens_q, hidden_states.device)
        elif position_ids is None and not is_packed: # Standard padded case
            kv_seq_len_ro = q_len # RoPE is applied on current q_len, not full kv_seq_len with cache
            if past_key_value is not None:
                 # This is tricky. If past_key_value exists, position_ids should be offset.
                 # Standard HF Llama does this:
                 # position_ids = torch.arange(past_key_value[0].shape[-2], kv_seq_len, ...)
                 # However, inplace_rope_embedding might expect position_ids relative to current query/key.
                 # Unsloth's original LlamaAttention forward pass uses:
                 # position_ids = torch.arange(past_key_value[0].shape[-2] if past_key_value is not None else 0, kv_seq_len ...)
                 # This seems correct for combined KV length.
                 # For packed, it's simpler as it's absolute within each sequence.
                past_kv_len_val = past_key_value[0].shape[-2]
                current_kv_len = key_states.shape[-2] # This is q_len for current K
                position_ids = torch.arange(past_kv_len_val, past_kv_len_val + current_kv_len, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0) #.expand(bsz, current_kv_len) - not needed as inplace_rope_embedding handles it

        # Apply RoPE. For packed, Q/K are (total_tokens, num_heads, head_dim)
        # For padded, Q/K are (bsz, num_heads, q_len, head_dim)
        # inplace_rope_embedding must handle both. It does if position_ids are correct.
        # It expects Q/K to be 3D (T, H, D) or 4D (B, H, S, D).
        # If packed, query_states is (total_tokens, num_attention_heads, head_dim)
        # position_ids is (total_tokens)
        # This should work with inplace_rope_embedding's logic.
        query_states, key_states = inplace_rope_embedding(
            query_states, key_states, position_ids, self.rotary_emb.sin_cached, self.rotary_emb.cos_cached
        )

        if not is_packed and past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache and not is_packed else None

        if not is_packed:
            # GQA/MQA repeat KV heads if necessary for padded inputs
            if self.num_key_value_groups > 1:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
            # Transpose for FA/Xformers/SDPA compatibility if not packed
            # FA/XF expect (bsz, q_len, num_heads, head_dim)
            # SDPA expects (bsz, num_heads, q_len, head_dim)
            query_states_for_attn = query_states.transpose(1, 2)
            key_states_for_attn = key_states.transpose(1, 2)
            value_states_for_attn = value_states.transpose(1, 2)
            # For SDPA, these need to be transposed back if FA/XF are not used.
        else:
            # For packed inputs, Q/K/V are already (total_tokens, num_heads, head_dim)
            # GQA/MQA for packed: repeat K/V along num_heads dimension before FA/XF/SDPA
            if self.num_key_value_groups > 1:
                # Original shape: (total_tokens, num_kv_heads, head_dim)
                # Target shape: (total_tokens, num_q_heads, head_dim)
                # This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)
                key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

            query_states_for_attn = query_states
            key_states_for_attn = key_states
            value_states_for_attn = value_states


        attn_output: torch.Tensor
        attn_weights: Optional[torch.Tensor] = None

        # Priority: Flash Attention > Xformers > SDPA
        can_use_flash_xf = attention_mask is None or is_packed # Packed inputs often don't use separate attention_mask

        if not output_attentions and HAS_FLASH_ATTENTION and can_use_flash_xf:
            attn_output = flash_attn_func(
                query_states_for_attn, key_states_for_attn, value_states_for_attn,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                causal=True, # Handles both packed (block-causal) and padded (causal)
            )
        elif not output_attentions and HAS_XFORMERS and BlockDiagonalCausalMask is not None and can_use_flash_xf :
            if is_packed:
                q_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
                kv_seqlens = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist() if cu_seqlens_kv is not None else q_seqlens
                attn_bias_for_xformers = BlockDiagonalCausalMask.from_seqlens(q_seqlens, kv_seqlens, is_causal=True)
            else: # Padded
                 # For padded, if attention_mask is None, it's causal. Xformers handles causal via LowerTriangularMask if attn_bias is not set.
                 # If attention_mask is provided (e.g. for padding), it's passed.
                 # The `xformers_attention` util currently takes `attention_mask` directly.
                 # For direct xops calls, we might need to construct the bias explicitly for causal.
                 # However, `can_use_flash_xf` implies `attention_mask is None` for padded, so causal is implied.
                 attn_bias_for_xformers = xops.fmha.attn_bias.LowerTriangularMask() if attention_mask is None else attention_mask

            if is_inference_pass and _xformers_memory_efficient_attention_forward_only is not None:
                attn_output = _xformers_memory_efficient_attention_forward_only(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn, attn_bias=attn_bias_for_xformers
                )
            else:
                # Default to the standard xformers op (xformers_attention is xops.memory_efficient_attention)
                attn_output = xops.memory_efficient_attention(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn, attn_bias=attn_bias_for_xformers
                )
        else:
            if is_packed:
                # SDPA with packed sequences needs a 2D mask (total_q_tokens, total_kv_tokens)
                # or rely on SDPA's internal support if Q,K,V are `NestedTensor` (requires PyTorch 2.1+)
                # For now, generate the mask manually.
                sdpa_mask = _create_block_diag_causal_sdpa_mask(
                    cu_seqlens_q, cu_seqlens_kv, total_tokens_q, total_tokens_kv,
                    dtype=query_states_for_attn.dtype, device=query_states_for_attn.device
                )
                # SDPA expects (total_tokens, num_heads, head_dim)
                # query_states_for_attn is already in this format for packed.
                attn_output = scaled_dot_product_attention(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn,
                    attn_mask=sdpa_mask,
                    is_causal=False, # Mask handles causality
                    dropout_p=0.0,
                )
            else: # Padded SDPA
                # Transpose back QKV for SDPA's expected (bsz, num_heads, q_len, head_dim)
                query_states_sdpa = query_states_for_attn.transpose(1, 2)
                key_states_sdpa = key_states_for_attn.transpose(1, 2)
                value_states_sdpa = value_states_for_attn.transpose(1, 2)

                # `attention_mask` for SDPA should be (bsz, num_heads, q_len, kv_seq_len) or broadcastable
                # Standard causal mask if attention_mask is None and q_len > 1
                attn_output = scaled_dot_product_attention(
                    query_states_sdpa, key_states_sdpa, value_states_sdpa,
                    attn_mask=attention_mask, # Use the original attention_mask for padded
                    is_causal=attention_mask is None and q_len > 1,
                    dropout_p=0.0,
                )
                attn_output = attn_output.transpose(1, 2) # Back to (bsz, q_len, num_heads, head_dim)

        # Reshape and project output
        # attn_output is (bsz, q_len, num_heads, head_dim) for padded
        # or (total_tokens, num_heads, head_dim) for packed
        if is_packed:
            attn_output = attn_output.reshape(total_tokens_q, self.num_attention_heads * self.head_dim)
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class MistralAttention(BaseAttention):
    def __init__(self, config, **kwargs):
        super().__init__(config, model_name="mistral", **kwargs)

        self.hidden_size = config.hidden_size
        # num_attention_heads, num_key_value_heads, head_dim, num_key_value_groups initialized in BaseAttention

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.sliding_window = getattr(config, "sliding_window", None)
        if self.sliding_window is not None and self.sliding_window <= 0:
            # Treat 0 or negative as no sliding window (infinite)
            self.sliding_window = None

        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048) # Mistral 7b uses 32768
        self.rope_theta = getattr(config, "rope_theta", 10000.0) # Mistral 7b uses 10000, some variants 1M for GGUF
        self.rope_scaling = getattr(config, "rope_scaling", None) # For compatibility if models use it
        self._init_rope()

    def _init_rope(self):
        # Mistral primarily uses LlamaRotaryEmbedding or its linear scaled version.
        # YaRN and LongRope are less common for official Mistral releases but might be used by fine-tunes.
        # For now, stick to LlamaRotary & LlamaLinearScalingRotaryEmbedding
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.rope_scaling.get("type")
            scaling_factor = self.rope_scaling.get("factor")
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    scaling_factor=scaling_factor,
                )
            # Add other RoPE types here if Mistral variants start using them more broadly.
            # For example, if YaRN becomes common for Mistral:
            # elif scaling_type == "yarn":
            #     self.rotary_emb = LlamaYaRNScalingRotaryEmbedding(...)
            else:
                # Default to LlamaRotaryEmbedding if scaling type is unknown or not 'linear'
                # Or raise error: raise ValueError(f"Unknown RoPE scaling type {scaling_type} for Mistral")
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )

    def _make_sliding_window_causal_mask(
        self,
        bsz: int,
        q_len: int,
        kv_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.sliding_window is None:
            # No sliding window, standard causal mask behavior (or None if q_len=1 for SDPA)
            if q_len > 1:
                # Standard causal mask. SDPA handles this with is_causal=True if mask is None.
                # However, to be explicit or for other backends, one could generate it.
                # For SDPA, returning None and setting is_causal=True is preferred.
                return None # Let SDPA handle it
            return None # No mask needed for q_len = 1

        # Create a causal mask with sliding window
        # Mask has shape (bsz, 1, q_len, kv_seq_len) for broadcasting to (bsz, n_heads, q_len, kv_seq_len)
        mask = torch.full((bsz, 1, q_len, kv_seq_len), dtype=dtype, device=device, value=torch.finfo(dtype).min)

        # Allow attention to current token and previous (sliding_window - 1) tokens
        # Also ensure causality: a query token cannot attend to future key tokens
        for i in range(q_len): # Query sequence index
            # Attachable key region: [max(0, i - (sliding_window-1)), i] within the *current* query's view of keys
            # But kv_seq_len includes past_kv. So, actual indices in K are different.
            # Let current_kv_start_idx = kv_seq_len - q_len
            # So, query i corresponds to key j = current_kv_start_idx + i
            # The valid key region for query i is [max(0, j - (sliding_window-1)), j]

            # Simpler: for each query token q_idx (0 to q_len-1)
            # It can attend to key tokens from max(0, kv_seq_len - q_len + q_idx - self.sliding_window + 1)
            # up to (kv_seq_len - q_len + q_idx).
            # This corresponds to positions in the combined K/V cache.

            # Correct indices for the mask relative to query position q_idx
            # start_key_idx = max(0, q_idx - (self.sliding_window - 1)) # This is if kv_seq_len == q_len
            # If past_kv exists, q_idx is relative to current query tokens
            # Keys are indexed from 0 to kv_seq_len - 1
            # Query q_idx (0-indexed in current batch) corresponds to "absolute" position:
            # effective_q_idx = (kv_seq_len - q_len) + q_idx

            # Valid key indices for query at effective_q_idx are:
            # [max(0, effective_q_idx - self.sliding_window + 1), effective_q_idx]

            # The mask is (bsz, 1, q_len, kv_seq_len)
            # mask[b, 0, q_token_idx, k_token_idx]
            # q_token_idx is the row, k_token_idx is the column

            # For each query position `row` (0 to q_len - 1)
            # The "time" of this query token is `query_time = past_kv_len + row`
            # It can attend to keys from `max(0, query_time - self.sliding_window + 1)` up to `query_time`
            # These are absolute positions in the combined KV sequence.
            past_kv_len = kv_seq_len - q_len
            min_attend_pos = torch.arange(q_len, device=device) + past_kv_len - self.sliding_window + 1
            min_attend_pos = torch.max(min_attend_pos, torch.zeros_like(min_attend_pos))

            # Create a grid for comparison
            col_indices = torch.arange(kv_seq_len, device=device).unsqueeze(0) # (1, kv_seq_len)
            row_indices = (torch.arange(q_len, device=device) + past_kv_len).unsqueeze(1) # (q_len, 1)

            # Valid if col_indices <= row_indices (causal)
            causal_cond = col_indices <= row_indices
            # Valid if col_indices >= min_attend_pos for that row
            sliding_cond = col_indices >= min_attend_pos.unsqueeze(1)

            combined_cond = causal_cond & sliding_cond
            mask[:, :, :, :] = torch.where(combined_cond.unsqueeze(0).unsqueeze(0), 0.0, torch.finfo(dtype).min)
        return mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # This is for padding mask primarily
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        is_inference_pass: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        is_packed_input = cu_seqlens_q is not None

        if packed_qkv is not None and use_cache:
            raise NotImplementedError("KV caching is not supported when `packed_qkv` is provided.")
        if packed_qkv is not None and past_key_value is not None:
             raise ValueError("`past_key_value` cannot be used when `packed_qkv` is provided.")

        if is_packed_input:
            if use_cache:
                raise NotImplementedError("KV caching is not supported for packed sequences.")
            if past_key_value is not None:
                raise ValueError("`past_key_value` cannot be used with packed sequences.")
            if self.sliding_window is not None and kwargs.get("padding_mask", None) is not None and packed_qkv is None:
                 pass

            total_tokens_q = hidden_states.shape[0] if packed_qkv is None else packed_qkv.shape[0]
            bsz, q_len = None, None
        else: # Padded
            if packed_qkv is not None:
                bsz, q_len = packed_qkv.shape[0], packed_qkv.shape[1]
            elif hidden_states is not None:
                bsz, q_len, _ = hidden_states.shape
            else:
                raise ValueError("Either hidden_states or packed_qkv must be provided for padded input.")
            total_tokens_q = bsz * q_len


        if packed_qkv is not None:
            q_slice_end = self.num_attention_heads
            k_slice_end = self.num_attention_heads + self.num_key_value_heads

            if is_packed_input:
                query_states = packed_qkv[:, :q_slice_end, :]
                key_states   = packed_qkv[:, q_slice_end:k_slice_end, :]
                value_states = packed_qkv[:, k_slice_end:, :]
            else: # Padded
                query_states = packed_qkv[:, :, :q_slice_end, :]
                key_states   = packed_qkv[:, :, q_slice_end:k_slice_end, :]
                value_states = packed_qkv[:, :, k_slice_end:, :]
                query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()
                key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

            if cu_seqlens_kv is None and is_packed_input: cu_seqlens_kv = cu_seqlens_q
            if max_seqlen_kv is None and is_packed_input: max_seqlen_kv = max_seqlen_q
            total_tokens_kv = total_tokens_q if is_packed_input else None
        else:
            if hidden_states is None:
                raise ValueError("`hidden_states` must be provided if `packed_qkv` is not.")
            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if is_packed_input:
                query_states = query_states.view(total_tokens_q, self.num_attention_heads, self.head_dim)
                key_states   = key_states.view(total_tokens_q, self.num_key_value_heads, self.head_dim)
                value_states = value_states.view(total_tokens_q, self.num_key_value_heads, self.head_dim)
                if cu_seqlens_kv is None: cu_seqlens_kv = cu_seqlens_q
                if max_seqlen_kv is None: max_seqlen_kv = max_seqlen_q
                total_tokens_kv = total_tokens_q
            else: # Padded
                query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
                key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None and is_packed:
            position_ids = _get_position_ids_from_cu_seqlens(cu_seqlens_q, hidden_states.device)
        elif position_ids is None and not is_packed:
            current_kv_len_ro = key_states.shape[-2] # This is q_len for current K
            past_kv_len_val = past_key_value[0].shape[-2] if past_key_value is not None else 0
            position_ids = torch.arange(past_kv_len_val, past_kv_len_val + current_kv_len_ro, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)

        query_states, key_states = inplace_rope_embedding(
            query_states, key_states, position_ids, self.rotary_emb.sin_cached, self.rotary_emb.cos_cached
        )

        if not is_packed and past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Sliding window KV cache truncation (PADDED input only)
        if not is_packed and self.sliding_window is not None and key_states.shape[-2] > self.sliding_window:
            key_states = key_states[:, :, -self.sliding_window:, :]
            value_states = value_states[:, :, -self.sliding_window:, :]

        past_key_value = (key_states, value_states) if use_cache and not is_packed else None

        effective_kv_seq_len = key_states.shape[-2] if not is_packed else None # For padded SDPA mask

        if self.num_key_value_groups > 1:
            if is_packed:
                key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output: torch.Tensor
        attn_weights: Optional[torch.Tensor] = None

        # Prepare Q, K, V for attention functions
        if is_packed:
            query_states_for_attn = query_states
            key_states_for_attn = key_states
            value_states_for_attn = value_states
        else: # Padded: transpose to (bsz, q_len, num_heads, head_dim) for FA/XF
            query_states_for_attn = query_states.transpose(1, 2)
            key_states_for_attn = key_states.transpose(1, 2)
            value_states_for_attn = value_states.transpose(1, 2)

        can_use_flash_xf = (attention_mask is None and not is_packed) or is_packed # Packed inputs usually don't use HF attention_mask

        if not output_attentions and HAS_FLASH_ATTENTION and can_use_flash_xf:
            attn_output = flash_attn_func(
                query_states_for_attn, key_states_for_attn, value_states_for_attn,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                causal=True, # Handles both packed (block-causal) and padded (causal)
                # For Mistral + FA with sliding window on packed: FA does not directly support SWA on packed.
                # SWA must be handled by KV cache management (already done if use_cache was true)
                # or by a custom mask if not using KV cache (SDPA path).
                # If packed and sliding window, FA will do block-causal, not sliding window within blocks.
                # This matches behavior if KV cache isn't used or isn't truncated by window for packed.
            )
        elif not output_attentions and HAS_XFORMERS and BlockDiagonalCausalMask is not None and can_use_flash_xf:
            if is_packed:
                q_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
                kv_seqlens = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist() if cu_seqlens_kv is not None else q_seqlens
                attn_bias_for_xformers = BlockDiagonalCausalMask.from_seqlens(q_seqlens, kv_seqlens, is_causal=True)
                # Note: Sliding window for packed Xformers would require custom bias. Currently block-causal.
                # If self.sliding_window is not None, this path won't do SWA for packed XF.
            else: # Padded
                # If attention_mask is None (can_use_flash_xf is True), it implies causal.
                # Mistral SWA for Xformers (padded): if attention_mask is None and SWA active,
                # relies on KV cache truncation. Xformers itself gets causal mask.
                # If a specific SWA mask for Xformers is needed, it would be in `attention_mask`.
                attn_bias_for_xformers = xops.fmha.attn_bias.LowerTriangularMask() if attention_mask is None else attention_mask

            if is_inference_pass and _xformers_memory_efficient_attention_forward_only is not None:
                attn_output = _xformers_memory_efficient_attention_forward_only(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn, attn_bias=attn_bias_for_xformers
                )
            else:
                # Default to the standard xformers op
                attn_output = xops.memory_efficient_attention(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn, attn_bias=attn_bias_for_xformers
                )
        else: # SDPA path
            if is_packed:
                sdpa_mask = _create_block_diag_causal_sdpa_mask(
                    cu_seqlens_q, cu_seqlens_kv, total_tokens_q, total_tokens_kv,
                    dtype=query_states_for_attn.dtype, device=query_states_for_attn.device,
                    sliding_window=self.sliding_window # Pass sliding window to SDPA mask
                )
                attn_output = scaled_dot_product_attention(
                    query_states_for_attn, key_states_for_attn, value_states_for_attn, # Shapes (T, H, D)
                    attn_mask=sdpa_mask,
                    is_causal=False, # Mask handles all causality and SWA
                    dropout_p=0.0,
                )
            else: # Padded SDPA
                query_states_sdpa = query_states_for_attn.transpose(1, 2)
                key_states_sdpa = key_states_for_attn.transpose(1, 2) # These are already K,V after repeat_kv
                value_states_sdpa = value_states_for_attn.transpose(1, 2) # (B,H,S,D)

                final_sdpa_mask = attention_mask # User-provided padding mask
                if self.sliding_window is not None: # Padded SWA
                    sliding_causal_mask = self._make_sliding_window_causal_mask(
                        bsz, q_len, effective_kv_seq_len, # effective_kv_seq_len is K's seq len after caching and SWA trim
                        dtype=query_states_sdpa.dtype, device=query_states_sdpa.device
                    )
                    if sliding_causal_mask is not None:
                        if final_sdpa_mask is None: final_sdpa_mask = sliding_causal_mask
                        else: # Merge if both exist
                            if final_sdpa_mask.ndim == 2: final_sdpa_mask = final_sdpa_mask.unsqueeze(1).unsqueeze(2)
                            final_sdpa_mask = final_sdpa_mask + sliding_causal_mask

                attn_output = scaled_dot_product_attention(
                    query_states_sdpa, key_states_sdpa, value_states_sdpa,
                    attn_mask=final_sdpa_mask,
                    is_causal=(self.sliding_window is None and final_sdpa_mask is None and q_len > 1),
                    dropout_p=0.0,
                )
                attn_output = attn_output.transpose(1, 2) # Back to (bsz, q_len, num_heads, head_dim)

        # Reshape and project output
        if is_packed:
            attn_output = attn_output.reshape(total_tokens_q, self.num_attention_heads * self.head_dim)
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
pass
