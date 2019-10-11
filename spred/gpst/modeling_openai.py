# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT model."""
# pylint: disable=invalid-name, bad-continuation, missing-function-docstring
# pylint: disable=no-member

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from functools import reduce

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_openai import Block
from transformers.modeling_openai import OpenAIGPTPreTrainedModel

DEBUG = False

logger = logging.getLogger(__name__)


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    """
    Parameters
    ----------
    input_ids : ``torch.LongTensor``.
        Shape: ``(batch_size, sequence_length)``.
        Indices of input sequence tokens in the vocabulary.
        Indices can be obtained using :class:`pytorch_transformers.BPT2Tokenizer`.
        See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
        :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for
        details.
    position_ids : ``torch.LongTensor``, optional.
        Indices of positions of each input sequence tokens in the position embeddings.
        Selected in the range ``[0, config.max_position_embeddings - 1]``.
        Shape: ``(batch_size, sequence_length)``.
    token_type_ids : ``torch.LongTensor``, optional.
        A parallel sequence of tokens (can be used to indicate various portions of the
        inputs). The embeddings from these tokens will be summed with the respective
        token embeddings. Indices are selected in the vocabulary (unlike BERT which
        has a specific vocabulary for segment indices).
        Shape: ``(batch_size, sequence_length)``.
    attention_mask : ``torch.FloatTensor``, optional.
        Mask to avoid performing attention on padding token indices.
        Shape: ``(batch_size, sequence_length)``.
        Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED,
            ``0`` for MASKED tokens.
    head_mask : ``torch.FloatTensor``, optional.
        Mask to nullify selected heads of the self-attention modules.
        Shape: ``(num_heads,)`` or ``(num_layers, num_heads)``.
        Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**,
            ``0`` indicates the head is **masked**.

    Returns
    -------
    last_hidden_state : ``torch.FloatTensor``
        Sequence of hidden-states at the last layer of the model.
        Shape: ``(batch_size, sequence_length, hidden_size)``.
    hidden_states : ``List[torch.FloatTensor]``, optional.
        Hidden-states of the model at the output of each layer plus the initial
        embedding outputs. A list of ``torch.FloatTensor`` (one for the output of each
        layer + the output of the embeddings).
        Only returned when ``config.output_hidden_states=True``.
        Shape: ``(batch_size, sequence_length, hidden_size)``.
    attentions : ``List[torch.FloatTensor``, optional.
        Attentions weights after the attention softmax, used to compute the weighted
        average in the self-attention heads.
        A list of ``torch.FloatTensor`` (one for each layer).
        Only returned when ``config.output_attentions=True``.
        Shape: ``(batch_size, num_heads, sequence_length, sequence_length)``.
    """

    def __init__(self, config):
        super(OpenAIGPTModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.tokens_embed = nn.Embedding(config.input_dim, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)]
        )

        # Pre-encoding layer performs a dense encoding from ``vocab_size`` -> ``n_embd``.
        self.pre_encoding = nn.Linear(config.input_dim, config.n_embd, bias=True)
        self.post_decoding = nn.Linear(config.n_embd, config.input_dim, bias=True)

        self.init_weights()
        # self.tie_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        self.tokens_embed = self._get_resized_embeddings(
            self.tokens_embed, new_num_tokens
        )
        return self.tokens_embed

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def tie_weights(self):
        """
        Make sure we are sharing the input and output embeddings. Export to
        TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.pre_encoding, self.post_decoding)

    def forward(self, input_ids, position_ids=None, inputs_raw=None, head_mask=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(-1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Expand to hidden dimension (``vocab_size`` -> ``n_embd``).
        inputs_raw = self.pre_encoding(inputs_raw)

        # Prepare head mask if needed.
        # 1.0 in head_mask indicates we keep the head.
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = inputs_raw
        position_embeds = self.positions_embed(position_ids)
        token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    hidden_states.view(*output_shape),
                )

            outputs = block(hidden_states, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = (hidden_states.view(*output_shape),)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # Map dimensionality back to ``vocab_size``.
        outputs = (self.post_decoding(outputs[0]),) + outputs[1:]
        return outputs  # last hidden state, (all hidden states), (all attentions)


class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    """
    Parameters
    ----------
    labels : ``torch.LongTensor``, optional.
        Shape: ``(batch_size, sequence_length)``.
        Labels for language modeling.
        Note that the labels **are shifted** inside the model, i.e. you can set
        ``labels = input_ids``.
        Indices are selected in ``[-1, 0, ..., config.orderbook_depth - 1]``
        All labels set to ``-1`` are ignored (masked), the loss is only
        computed for labels in ``[0, ..., config.orderbook_depth - 1]``.

    Returns
    -------
    loss : ``torch.FloatTensor``, optional.
        Language modeling loss. Only returned when ``labels`` is provided.
        Shape: ``(1,)``.
    prediction_scores : ``torch.FloatTensor``.
        Prediction scores of the language modeling head (scores for each vocabulary
        token before SoftMax).
        Shape: ``(batch_size, sequence_length, config.orderbook_depth)``.
    hidden_states : ``List[torch.FloatTensor]``, optional.
        Hidden-states of the model at the output of each layer plus the initial
        embedding outputs. Only returned when ``config.output_hidden_states=True``.
        A list of ``torch.FloatTensor`` (one for the output of each layer + the output
        of the embeddings).
        Shape: ``(n_layers + 1, batch_size, sequence_length, hidden_size)``.
    attentions : ``List[torch.FloatTensor``, optional.
        Attentions weights after the attention softmax, used to compute the weighted
        average in the self-attention heads.
        Only returned when ``config.output_attentions=True``.
        Shape: ``(n_layers, batch_size, num_heads, sequence_length, sequence_length)``.
    """

    def __init__(self, config):
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.depth_range = 2 * config.orderbook_depth + 1
        self.orderbook_depth = config.orderbook_depth
        self.mode = config.mode
        self.bid_delta_head = nn.Linear(
            config.input_dim, config.orderbook_depth, bias=False
        )
        self.ask_delta_head = nn.Linear(
            config.input_dim, config.orderbook_depth * self.depth_range, bias=False
        )
        self.bid_classification_head = nn.Linear(config.input_dim, 3, bias=False)
        self.ask_classification_head = nn.Linear(
            config.input_dim, 3 * self.depth_range, bias=False
        )

        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        labels: torch.LongTensor,
        inputs_raw: torch.FloatTensor,
        head_mask: torch.FloatTensor = None,
    ):
        bsz, seq_len = input_ids.shape[:2]
        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,
            inputs_raw=inputs_raw,
            head_mask=head_mask,
        )
        hidden_states = transformer_outputs[0]
        assert hidden_states.shape == inputs_raw.shape

        if self.mode in ["bid_increase", "bid_decrease"]:
            logits = self.bid_delta_head(hidden_states)
        elif self.mode in ["ask_increase", "ask_decrease"]:
            logits = self.ask_delta_head(hidden_states)
            logits_matrix = logits.view(
                logits.shape[0], logits.shape[1], self.depth_range, self.orderbook_depth
            )
        elif self.mode == "bid_classification":
            logits = self.bid_classification_head(hidden_states)
        elif self.mode == "ask_classification":
            logits = self.ask_classification_head(hidden_states)
            logits_matrix = logits.view(
                logits.shape[0], logits.shape[1], self.depth_range, 3
            )
        else:
            print("Value of config param ``mode``: %s" % self.mode)
            raise ValueError("Config param ``mode`` is invalid.")

        outputs = (logits,) + transformer_outputs[1:]

        # Shape of ``labels``: ``(bsz, seq_len, 6)``.
        # Type of ``labels``: ``torch.LongTensor`` (integer values).
        # Range of values of ``labels``:
        #   ``1 <= label <= config.orderbook_depth`` or ``-1`` (mask) for F networks.
        if labels is not None:
            # Shift so that tokens < n predict n.
            if self.mode[:3] == "bid":
                assert labels.shape == (bsz, seq_len)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            elif self.mode[:3] == "ask":
                assert labels.shape == (bsz, seq_len, self.depth_range)
                shift_logits = logits_matrix[..., :-1, :, :].contiguous()
                shift_labels = labels[..., 1:, :].contiguous()
            else:
                # TODO: fix error message.
                raise ValueError("Mode has invalid value")
            # Flatten the tokens.
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss_input = shift_logits.view(-1, shift_logits.size(-1))
            loss_target = shift_labels.view(-1)
            loss = loss_fct(loss_input, loss_target)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (all hidden states), (all attentions)

    def regression_loss(self, hidden, labels):
        logits = hidden
        diff = logits - labels
        loss = torch.mul(diff, diff)

        # Make a scalar.
        loss = torch.mean(loss)

        return loss

    def smape(self, predicted, true):
        epsilon = 0.1
        device = true.device
        ones = torch.ones(true.shape).to(device)
        summ = torch.max(
            torch.abs(true) + torch.abs(predicted) + epsilon, 0.5 + epsilon * ones
        )
        smape = torch.abs(predicted - true) / summ * 2.0
        smape = torch.mul(smape, smape)
        smape_shape = smape.shape
        smape = torch.sum(smape) / reduce((lambda x, y: x * y), smape_shape)

        return smape
