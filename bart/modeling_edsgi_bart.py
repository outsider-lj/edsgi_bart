# coding=utf-8
import math
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
from modeling.activations import ACT2FN
from modeling.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from modeling.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from .modeling_edsgi_utils import PreTrainedModel
import logging
from .configuration_bart import BartConfig


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
]
# This list is incomplete. See all BART models at https://huggingface.co/models?filter=bart
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels#.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    """
    prev_output_tokens = input_ids.clone()
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, past_key_values_length: int = 0
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    if past_key_values_length > 0:
        # concat fully attendend attention_mask to the beginning if `past_key_values` are used
        expanded_mask = torch.cat(
            [
                torch.ones(bsz, 1, tgt_len, past_key_values_length, device=expanded_mask.device, dtype=dtype),
                expanded_mask,
            ],
            dim=-1,
        )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def BartLayerNorm(normalized_shape: torch.Size, eps: float = 1e-5, elementwise_affine: bool = True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size=None,position_ids:torch.Tensor=None, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if position_ids is not None:
            positions=position_ids
        else:
            bsz, seq_len = input_ids_shape[:2]
            positions = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            )
        return super().forward(positions + self.offset)


class BartSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_steps_mask:Optional[torch.Tensor] = None,
            decoder_step_states: Optional[torch.Tensor] = None,
            focused_attention: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
            hard_attention: bool = False,
            final_decoder: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # elif decoder_step_states is not None:
            # key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            # value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            # decoder_states_hidden_states=self._shape(decoder_steps_hidden_state,-1,bsz)
            # key_states = self._shape(self.k_proj(torch.cat([decoder_step_states, hidden_states], 1)), -1, bsz)
            # value_states = self._shape(self.v_proj(torch.cat([decoder_step_states, hidden_states], 1)), -1, bsz)
            # key_states = torch.cat([self._shape(decoder_steps_hidden_state,-1,bsz), key_states], dim=1)
            # value_states = torch.cat([decoder_steps_hidden_state, value_states], dim=1)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"


        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # if decoder_steps_mask is not None:
        #     attn_weights.mul(decoder_steps_mask)
        if focused_attention is not None:
            focused_attention=focused_attention[:,None,:, :].expand(bsz,self.num_heads,tgt_len,src_len).to(attn_weights.dtype)
            focused_attention=focused_attention.contiguous().view(bsz*self.num_heads,tgt_len,src_len)
            exp_attn_weight=focused_attention*torch.exp(attn_weights)
            sum_exp_attn_weight=torch.sum(exp_attn_weight,dim=-1)
            attn_weights=torch.div(exp_attn_weight,sum_exp_attn_weight.unsqueeze(2))
            # if hard_attention==True:
            #     attn_weights=gumbel_softmax(attn_weights)
            # attn_weights = focused_attention*attn_weights
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)
        # if decoder_steps_hidden_state is not None:
        #     print(attn_weights[1])
        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value



class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = BartLayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = BartLayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`): Whether the base model outputs attentions. This requires the attentions tensor to be reshaped in this function.
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, attn_weights


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.config=config
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = BartLayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        if config.decoder_step_att==True:
            self.decoder_step_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                )
        if config.decoder_focus_att==True:
            self.decoder_focus_attn=BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                )
        self.encoder_attn_layer_norm = BartLayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = BartLayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
            decoder_steps_mask: Optional[Tuple[torch.Tensor]] = None,
            decoder_step_states: Optional[Tuple[torch.Tensor]] = None,
            focused_attention: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[torch.Tensor] = False,
            final_decoder: Optional[torch.Tensor] = False,
            hard_attention: Optional[torch.Tensor] = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`): Whether the base model outputs attentions. This requires the attentions tensor to be reshaped in this function.
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            decoder_step_states=decoder_step_states,
            decoder_steps_mask=decoder_steps_mask,
            final_decoder=final_decoder
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        if decoder_step_states is not None:
            hidden_states=hidden_states
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.decoder_step_attn(
                hidden_states=hidden_states,
                key_value_states=decoder_step_states,
                attention_mask=decoder_steps_mask,
                # decoder_emo_states=decoder_emo_states,
                # decoder_topic_states=decoder_topic_states,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            if not self.normalize_before:
                step_hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
        # Cross-Attention Block

        # cross_attn_mask1=_expand_mask(attention_mask, encoder_attention_mask.dtype)
        # cross_attn_mask2=_expand_mask( encoder_attention_mask,attention_mask.dtype)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            if not self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
        # if focused_attention is not None:
        if self.config.decoder_focus_att==True:
            hidden_states=hidden_states
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.decoder_focus_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                focused_attention=focused_attention,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                hard_attention=hard_attention,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            if not self.normalize_before:
                step_hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
        # Cross-Attention Block

        # cross_attn_mask1=_expand_mask(attention_mask, encoder_attention_mask.dtype)
        # cross_attn_mask2=_expand_mask( encoder_attention_mask,attention_mask.dtype)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        # Fully Connected
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return (
            hidden_states,
            self_attn_weights,
            present_key_value,
            cross_attn_weights,
        )


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BartSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs




class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> # see ``examples/summarization/bart/run_eval.py`` for a longer example
        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the :obj:`input_ids` to the right, following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Module] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if config.static_position_embeddings:
            self.embed_positions = BartSinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = BartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = BartLayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = BartLayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
            position_ids=None,
            token_type_ids=None,
            emo_input_ids=None,
            vm=None,
        attention_mask=None,
        encoder_token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids,token_type_ids) * self.embed_scale
        if position_ids is not None:
            embed_pos = []
            for pos in position_ids:
                embed = self.embed_positions(position_ids=pos)
                embed_pos.append(embed)
            embed_pos = torch.stack(embed_pos)
        else:
            embed_pos = self.embed_positions(input_ids_shape=input_shape)
        # embed_pos.view(inputs_embeds.size())
        hidden_states = inputs_embeds + embed_pos#+token_type_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None and vm==None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        elif vm is not None:
            if len(vm.size())<4:
                attention_mask=vm.to(inputs_embeds.dtype).unsqueeze(1)
            else:
                attention_mask=vm.to(inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        encoder_hidden_states=[]
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                hidden_states, attn = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Module] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.do_blenderbot_90_layernorm = config.do_blenderbot_90_layernorm  # layernorm variant
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if config.static_position_embeddings:
            self.embed_positions = BartSinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = BartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = BartLayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = BartLayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
            token_type_ids=None,
            emo_input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
            final_decoder=False,
            decoder_steps_len=None,
            decoder_steps_mask=None,
            decoder_step_states=None,
            focused_attention=None,
            hard_attention=False,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if final_decoder is True and decoder_step_states is not None and input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.embed_tokens(input_ids, emo_input_ids) * self.embed_scale
            # inputs_embeds=torch.cat((decoder_last_states,inputs_embeds),dim=1)
            input_ids=None

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids,token_type_ids) * self.embed_scale
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 0:#1
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        # create decoder_padding_mask if not provided and needed
        # 4.12.20 (PVP): Not a fan of this "magical" function that
        # automatically creates attention_mask for padded tokens
        # => this is inconsistent with other models
        # => Pegasus uses the pad_token as decoder_start_token_id, so that this could
        # pose some problems.
        if (
            attention_mask is None
            and input_ids is not None
            and input_shape[-1] > 1
            and self.config.pad_token_id in input_ids
        ):
            # should be kept for backwards compatibility
            attention_mask = input_ids.ne(self.config.pad_token_id).to(torch.long)
            # never mask leading token, even if it is pad
            attention_mask[:, 0] = attention_mask[:, 1]

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None :
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # if final_decoder is True and decoder_steps_mask is not None:
        #     steps_attention_mask = _expand_mask(decoder_steps_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        #     combined_attention_mask= torch.cat((steps_attention_mask, combined_attention_mask), 3)
        if decoder_steps_mask is not None:
            steps_attention_mask = _expand_mask(decoder_steps_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        else:
            steps_attention_mask=None
        # embed positions
        positions = self.embed_positions(input_ids_shape=input_shape, past_key_values_length=past_key_values_length)
        #
        if self.do_blenderbot_90_layernorm:
            hidden_states = self.layernorm_embedding(inputs_embeds)
            hidden_states += positions
        else:
            hidden_states = inputs_embeds + positions
            hidden_states = self.layernorm_embedding(hidden_states)
        if final_decoder==True:
            hidden_states_emd=hidden_states.detach()
        else:
            hidden_states_emd=hidden_states
        # hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states_emd, p=self.dropout, training=self.training)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, layer_self_attn, present_key_value, layer_cross_attn = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                decoder_steps_mask=steps_attention_mask,
                decoder_step_states=decoder_step_states,
                focused_attention=focused_attention,
                hard_attention=hard_attention,
                # final_decoder=final_decoder
            )

            if use_cache:
                next_decoder_cache += (present_key_value,)

            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attentions += (layer_cross_attn,)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # if config.add_final_layer_norm (mBART)
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class StepBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        decoder_step_config = copy.copy(config)
        decoder_step_config.decoder_focus_att = True
        decoder_step_config.decoder_step_att = True
        self.embed_dim = config.d_model
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = MKEmbedding(config)
        self.encoder = BartEncoder(config, self.shared)
        self.decoder2= BartDecoder(decoder_step_config, self.shared)
        self.decoder1= BartDecoder(decoder_step_config,self.shared)
        # self.decoder = BartDecoder(decoder_final_config, self.shared)
        self.dropout=nn.Dropout2d(config.dropout)
        self.att_dropout=nn.Dropout(config.dropout)
        self.layer_norm = BartLayerNorm(config.d_model) #if config.add_final_layer_norm else None
        # if config.n_special>0:
            # self.special_tokens_embed=nn.Embedding(config.n_special,config.d_model, padding_idx)
            # self.encoder.embed_tokens = nn.Embedding(config.n_special+config.vocab_size, config.d_model, padding_idx)
        self.activation_fn = ACT2FN[config.activation_function]
        emo_att = torch.Tensor(1, config.hidden_size)
        topic_att = torch.Tensor(1, config.hidden_size)
        emo_att = torch.nn.init.uniform_(emo_att)
        topic_att = torch.nn.init.uniform_(topic_att)
        self.emo_attention_vector = nn.Parameter(emo_att)
        self.topic_attention_vector = nn.Parameter(topic_att)
        self.init_weights()
        # layer=self.encoder.layers[-1]
        # self._tie_encoder_decoder_weights(self.emo_attn,self.encoder.layers[-1],base_model_prefix="")

    def get_input_embeddings(self):
        return self.shared.embed_token

    # def set_input_embeddings(self, value):
    #     self.shared.embed_token = value
    #     self.encoder.embed_tokens = self.shared
    #     self.decoder.embed_tokens = self.shared
    #     self.decoder2.embed_tokens = self.shared
    #     self.decoder1.embed_tokens=self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    def set_num_special_tokens(self, special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        num_special_tokens=len(special_tokens)
        self.config.n_special = num_special_tokens
        self.special_tokens_embed = nn.Embedding(self.config.n_special, self.config.d_model)
        # self.model.shared.num_embeddings=
        # Build new embeddings and initialize all new embeddings (in particular the special tokens)
        old_embed = self.encoder.embed_tokens
        torch.nn.init.normal_(self.special_tokens_embed.weight,mean=0,std=0.02)
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.d_model,padding_idx=self.config.pad_token_id)
        self.encoder.embed_tokens = nn.Embedding(self.config.total_tokens_embeddings, self.config.d_model,padding_idx=self.config.pad_token_id)
        self.encoder.embed_tokens.to(old_embed.weight.device)
        # self.init_weights()
        # Copy word embeddings from the previous weights
        self.encoder.embed_tokens.weight.data[:self.config.vocab_size, :] = old_embed.weight.data[:self.config.vocab_size, :]
        self.encoder.embed_tokens.weight.data[self.config.vocab_size:, :]=self.special_tokens_embed.weight.data

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
            token_type_ids=None,
            situation_ids=None,
            situation_mask=None,
            emo_intensity=None,
            knowledge_confidence=None,
        position_ids=None,
            vm=None,
        attention_mask=None,
        encoder_token_type_ids=None,
        decoder_input_ids_first=None,
        decoder_input_ids_second=None,
        decoder_input_ids_final=None,
        emo_input1_ids=None,
        emo_input2_ids=None,
        emo_inputfinal_ids=None,
        decoder_attention_mask_first=None,
        decoder_attention_mask_second=None,
        decoder_attention_mask_final=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
            emotion_focused_attention=None,
            topic_focused_attention=None,
        return_dict=None,
            is_train=True,
            is_integrate=True,
            hard_attention=False,
    ):

        # 4.12.20 (PVP): Not a fan of this "magical" function and
        # also wonder how often it's actually used ... keep now
        # for backward compatibility
        # -> is this used for backward compatibility
        if decoder_input_ids_first is None and decoder_inputs_embeds is None:
            decoder_input_ids_first = shift_tokens_right(input_ids, self.config.pad_token_id)
        if decoder_input_ids_second is None and decoder_inputs_embeds is None:
            decoder_input_ids_second = shift_tokens_right(input_ids, self.config.pad_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # encoder_inputs_embeds=self.shared(input_ids,emo_input_ids)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                vm=vm,
                attention_mask=attention_mask,
                encoder_token_type_ids=encoder_token_type_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        #获取情感特征
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_states = encoder_hidden_states
        inverted_mask = 1.0 - attention_mask  # .unsqueeze(1)
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(encoder_states.dtype).min)
        emo_att = torch.matmul(self.emo_attention_vector,
                               encoder_states.transpose(1, 2)) + inverted_mask.unsqueeze(1)
        emotion_focused_attention = nn.functional.softmax(emo_att,-1)  # torch.mul(attention_mask.unsqueeze(1), emo_att)
        emo_att = self.att_dropout(emotion_focused_attention)
        emo_feature = torch.matmul(emo_att, encoder_states)
        # topic_att = torch.matmul(self.topic_attention_vector,
        #                          encoder_states.transpose(1, 2)) + inverted_mask.unsqueeze(1)
        # topic_focused_attention = nn.functional.softmax(topic_att,dim=-1)  # torch.mul(situation_mask.unsqueeze(1), topic_att)
        # topic_focused_attention = self.att_dropout(topic_focused_attention)
        # topic_feature = torch.matmul(topic_focused_attention, encoder_states)
        # topic_feature = nn.functional.normalize(topic_feature, dim=-1)
        #获取话题特征
        if hard_attention==True:
            knowledge_confidence=torch.eq(knowledge_confidence,1).to(emotion_focused_attention.dtype)
            knowledge_confidence_mask=1.0-knowledge_confidence
            knowledge_confidence_mask=knowledge_confidence_mask.masked_fill(knowledge_confidence_mask.bool(), torch.finfo(emotion_focused_attention.dtype).min)
            emo_intensity_mask=torch.eq(emo_intensity,0).to(emotion_focused_attention.dtype)
            emo_intensity_mask=emo_intensity_mask.masked_fill(emo_intensity_mask.bool(),torch.finfo(emotion_focused_attention.dtype).min)
            emo_intensity=gumbel_softmax(emo_intensity,hard=False)+emo_intensity_mask
            knowledge_confidence=gumbel_softmax(knowledge_confidence,hard=False)+knowledge_confidence_mask
            knowledge_confidence=nn.functional.softmax(knowledge_confidence,dim=-1)
            # emo_intensity_sum=torch.sum(emo_intensity,dim=-1)
            # emo_intensity=torch.div(emo_intensity,emo_intensity_sum)
            # emo_intensity=emo_intensity+emo_intensity_mask
            emo_intensity=nn.functional.softmax(emo_intensity,dim=-1)
            emotion_focused_attention=emo_intensity
            topic_focused_attention = knowledge_confidence
            # emotion_focused_attention=torch.div((emo_intensity+emotion_focused_attention),2)
            # topic_focused_attention = torch.div((knowledge_confidence + emotion_focused_attention), 2)

        if situation_ids is not None:
            situation_token_type_ids=torch.full(situation_ids.size(),50265).to(situation_ids.device)
            situation_outputs = self.encoder(
                situation_ids,
                attention_mask=situation_mask,
                token_type_ids=situation_token_type_ids,
                position_ids=None,
                # head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = situation_outputs[0]
            # pooled_output = self.dropout(pooled_output)
            # situation_feature=pooled_output[:, 1, :]
            situation_inverted_mask = 1.0 - situation_mask  # .unsqueeze(1)
            situation_inverted_mask = situation_inverted_mask.masked_fill(situation_inverted_mask.bool(),
                                                                          torch.finfo(sequence_output.dtype).min)
            situation_att = torch.matmul(self.topic_attention_vector,
                                         sequence_output.transpose(1, 2)) + situation_inverted_mask.unsqueeze(1)
            situation_att = nn.functional.softmax(situation_att, dim=-1)
            # situation_att=nn.functional.dropout(situation_att,p=0.1)
            situation_feature = torch.matmul(situation_att, sequence_output)
            situation_feature = nn.functional.normalize(situation_feature, dim=-1)
            topic_similarity = torch.nn.CosineSimilarity(dim=-1)(topic_feature, situation_feature)
        else :
            topic_similarity=None
        # input_shape=input_ids.size()
        if attention_mask==None:
            attention_mask_=None
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask_ = _expand_mask(attention_mask, encoder_outputs[0].dtype)

        decoder_inputs1_embeds = self.shared(decoder_input_ids_first, None)
        decoder_outputs_first = self.decoder1(
            input_ids=None,#decoder_input_ids_first,
                emo_input_ids=emo_input1_ids,
            attention_mask=decoder_attention_mask_first,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs1_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            focused_attention=emotion_focused_attention,
                hard_attention=hard_attention,
            return_dict=return_dict,
            )
        decoder_inputs2_embeds = self.shared(decoder_input_ids_second, None)
        decoder_outputs_second = self.decoder2(
            input_ids=None,#decoder_input_ids_second,
                emo_input_ids=emo_input2_ids,
            attention_mask=decoder_attention_mask_second,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs2_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
                focused_attention=topic_focused_attention,
                hard_attention=hard_attention,
            return_dict=return_dict,
            )
        last_hidden_state_first = decoder_outputs_first[0]
        last_hidden_state_second = decoder_outputs_second[0]
        if is_train==True:
            if is_integrate==False:
                return (last_hidden_state_first, last_hidden_state_second,emo_feature,topic_similarity)
            else:
                return (last_hidden_state_first, last_hidden_state_second, encoder_hidden_states, emo_feature,topic_similarity)
        else:
            if is_integrate==False:
                return (last_hidden_state_first, last_hidden_state_second,emo_feature)
            else:
                return (last_hidden_state_first, last_hidden_state_second, encoder_hidden_states, emo_feature, topic_similarity)

@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class StepBartForDialogueGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        config.decoder_focus_att = False
        config.step_focus_att = False
        self.model = StepBartModel(config)
        decoder_final_config = copy.copy(config)
        decoder_final_config.decoder_step_att = True
        self.integration=BartDecoder(decoder_final_config, self.model.shared)
        num_embeddings=self.model.shared.embed_token.num_embeddings
        self.register_buffer("final_logits_bias", torch.zeros((1,num_embeddings)))
        self.lm_head_first = nn.Linear(config.d_model, num_embeddings, bias=False)
        self.lm_head_second = nn.Linear(config.d_model,num_embeddings, bias=False)
        self.lm_head = nn.Linear(config.d_model, num_embeddings, bias=False)
        # self.emo_head = nn.Linear(config.d_model, 32, bias=False)
        self.emo_head = BartClassificationHead(input_dim=config.hidden_size,
                                                         inner_dim=config.hidden_size,
                                                         num_classes=32,
                                                         pooler_dropout=config.dropout)
        # self.emotion_choice_head = BARTEmotionChoiceHead(config)
        self.init_weights()


    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    def get_input_embeddings(self) :
        return self.model.shared.embed_token
    def get_output_embeddings(self):
        return self.lm_head
    def get_output_embeddings_first(self):
        return self.lm_head_first
    def get_output_embeddings_second(self):
        return self.lm_head_second
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def set_input_embeddings(self, value):
        self.model.shared.embed_token = value
        self.model.encoder.embed_tokens = self.model.shared
        self.integration.embed_tokens = self.model.shared
        self.model.decoder2.embed_tokens = self.model.shared
        self.model.decoder1.embed_tokens=self.model.shared
    # def get_convert(self):
    #     return self.model.convert#,self.model.convert2)

    def set_num_special_tokens(self,special_tokens):
        self.model.set_num_special_tokens(special_tokens)
        # self.lm_head= nn.Linear(self.config.d_model, self.model.shared.num_embeddings, bias=False)

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
            token_type_ids=None,
            situation_ids=None,
            situation_mask=None,
            emo_intensity=None,
            knowledge_confidence=None,
        position_ids=None,
            vm=None,
        attention_mask=None,
        encoder_token_type_ids=None,
        decoder_input_ids_first=None,
        decoder_input_ids_second=None,
        decoder_input_ids_final=None,
            emo_input1_ids=None,
            emo_input2_ids=None,
            emo_inputfinal_ids=None,
        decoder_attention_mask_first=None,
        decoder_attention_mask_second=None,
        decoder_attention_mask_final=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        emotion_focused_attention=None,
        topic_focused_attention=None,
        return_dict=None,
            is_train=True,
            is_integrate=False,
            hard_attention=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> # Mask filling only works for bart-large
            >>> from transformers import BartTokenizer, BartForConditionalGeneration
            >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
            >>> # ['good', 'great', 'all', 'really', 'very']
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids_first is None:
                decoder_input_ids_first = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            situation_ids=situation_ids,
            situation_mask=situation_mask,
            emo_intensity=emo_intensity,
            knowledge_confidence=knowledge_confidence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            vm=vm,
            encoder_token_type_ids=encoder_token_type_ids,
            decoder_input_ids_first=decoder_input_ids_first,
            decoder_input_ids_second=decoder_input_ids_second,
            decoder_input_ids_final=decoder_input_ids_final,
            emo_input1_ids=emo_input1_ids,
            emo_input2_ids=emo_input2_ids,
            emo_inputfinal_ids=emo_inputfinal_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask_first=decoder_attention_mask_first,
            decoder_attention_mask_second=decoder_attention_mask_second,
            decoder_attention_mask_final=decoder_attention_mask_final,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            emotion_focused_attention=emotion_focused_attention,
            topic_focused_attention=topic_focused_attention,
            is_train=is_train,
            is_integrate=is_integrate,
            hard_attention=hard_attention
        )
        if is_train==True:
            if is_integrate==False:
                outputs_first, outputs_second ,emo_feature,topic_similarity= outputs

                lm_logits_first = self.lm_head_first(outputs_first)  # + self.final_logits_bias
                lm_logits_second = self.lm_head_second(outputs_second)
                emo_logits=self.emo_head(emo_feature)
                # emo_logits = self.emotion_choice_head(encoder_hidden_states, mc_token_ids)
                return lm_logits_first, lm_logits_second,emo_logits,topic_similarity
            else:
                outputs_first, outputs_second,encoder_hidden_states,emo_feature ,topic_similarity= outputs
                with torch.no_grad():
                    emotion_feedback =outputs_first.detach()
                    topic_discussion = outputs_second.detach()
                    encoder_states = encoder_hidden_states.detach()
                # decoder_step_states = self.convert(self.dropout(torch.cat((emotion_feedback, topic_discussion), 1)))
                decoder_step_states = torch.cat((emotion_feedback, topic_discussion), 1)
                # decoder_step_states = self.layer_norm(decoder_step_states)
                # decoder_step_states=torch.cat((decoder_emo_states,decoder_topic_states),dim=1)
                decoder_steps_mask = torch.cat((decoder_attention_mask_first, decoder_attention_mask_second), dim=1)
                decoder_final_inputs_embeds = self.model.shared(decoder_input_ids_final, None)  # * self.embed_scale
                decoder_outputs = self.integration(
                    input_ids=None,  # decoder_input_ids_final,
                    # emo_input_ids=emo_inputfinal_ids,
                    attention_mask=decoder_attention_mask_final,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_final_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    final_decoder=True,
                    decoder_step_states=decoder_step_states,
                    decoder_steps_mask=decoder_steps_mask
                    # decoder_steps_len=decoder_steps_len,
                )
                outputs_final=decoder_outputs[0]
                lm_logits_first = self.lm_head_first(outputs_first)  # + self.final_logits_bias
                lm_logits_second = self.lm_head_second(outputs_second)
                lm_logits_final = self.lm_head(outputs_final)
                emo_logits = self.emo_head(emo_feature)
                return lm_logits_first, lm_logits_second,lm_logits_final,emo_logits,topic_similarity
        else:
            if is_integrate==False:
                outputs_first, outputs_second, emo_feature = outputs
                lm_logits_first = self.lm_head_first(outputs_first)  # + self.final_logits_bias
                lm_logits_second = self.lm_head_second(outputs_second)
                # lm_logits_final = self.lm_head(outputs_final)
                emo_logits = self.emo_head(emo_feature)
                return lm_logits_first, lm_logits_second,emo_logits, outputs_first, outputs_second
            else:
                outputs_first, outputs_second, encoder_hidden_states, emo_feature, topic_similarity = outputs
                encoder_states = encoder_hidden_states  # .detach()
                # decoder_step_states = self.convert(self.dropout(torch.cat((emotion_feedback, topic_discussion), 1)))
                # decoder_step_states = torch.cat((outputs_first, outputs_second), 1)
                # decoder_step_states = self.layer_norm(decoder_step_states)
                decoder_emo_states, decoder_topic_states=decoder_inputs_embeds
                decoder_step_states=torch.cat((decoder_emo_states,decoder_topic_states),dim=1)
                decoder_steps_mask = torch.cat((decoder_attention_mask_first, decoder_attention_mask_second), dim=1)
                decoder_final_inputs_embeds = self.model.shared(decoder_input_ids_final, None)  # * self.embed_scale
                decoder_outputs = self.integration(
                    input_ids=None,  # decoder_input_ids_final,
                    # emo_input_ids=emo_inputfinal_ids,
                    attention_mask=decoder_attention_mask_final,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=decoder_final_inputs_embeds,  # decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    final_decoder=True,
                    decoder_step_states=decoder_step_states,
                    decoder_steps_mask=decoder_steps_mask
                    # decoder_steps_len=decoder_steps_len,
                )
                outputs_final=decoder_outputs[0]
                lm_logits_final = self.lm_head(outputs_final)
                return  lm_logits_final#,outputs



    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

class BARTEmotionChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(BARTEmotionChoiceHead, self).__init__()
        self.n_embd = config.d_model
        self.dropout = nn.Dropout2d(config.dropout)  # To reproduce the noise_shape parameter of TF implementation
        num_emotions = 32
        self.linear = nn.Linear(config.d_model, num_emotions)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, seq_length, hidden_size)
        # mc_token_ids (bsz,)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        # mc_token_ids (bsz, 1, hidden_size)
        multiple_choice_h = hidden_states.gather(1, mc_token_ids).squeeze(1)
        # multiple_choice_h (bsz, hidden_size)
        multiple_choice_h = self.dropout(multiple_choice_h)
        multiple_choice_logits = self.linear(multiple_choice_h)
        # (bsz, num_choices)
        return multiple_choice_logits

class MKEmbedding(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(MKEmbedding, self).__init__()
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # emo_cls=9
        embed_dim=config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_token = nn.Embedding(vocab_size, config.d_model, padding_idx)
        # self.emo_embedding=nn.Embedding(emo_cls, config.d_model, 8)
        # self.emo_embedding.weight.data.normal_(mean=0.0, std=config.init_std)
        # if self.emo_embedding.padding_idx is not None:
        #     self.emo_embedding.weight.data[self.emo_embedding.padding_idx].zero_()

    def forward(self,input_ids=None,token_type_ids=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_token(input_ids) * self.embed_scale
        # if position_ids is not None:
        #     embed_pos = []
        #     for pos in position_ids:
        #         embed = self.embed_positions(position_ids=pos)
        #         embed_pos.append(embed)
        #     embed_pos = torch.stack(embed_pos)
        # else:
        #     embed_pos = self.embed_positions(input_ids_shape=input_shape)
        if token_type_ids is not None:
            token_type_embeds=self.embed_token(token_type_ids)
        else:
            token_type_ids=torch.full(input_shape,50266).to(input_ids.device)
            token_type_embeds=self.embed_token(token_type_ids)
        # if emo_input_ids is None:
        #     embeds=inputs_embeds#+embed_pos
        # else:
        #     embed_emo = self.emo_embedding(emo_input_ids)
        embeds=inputs_embeds+token_type_embeds
        return embeds