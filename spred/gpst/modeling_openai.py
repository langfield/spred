# coding=utf-8
""" PyTorch OpenAI GPT model. """
# pylint: disable=bad-continuation, missing-function-docstring, no-member
# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_openai import OpenAIGPTConfig
from transformers.modeling_openai import OpenAIGPTPreTrainedModel, Block

LOGGER = logging.getLogger(__name__)


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

    def __init__(self, config: OpenAIGPTConfig) -> None:
        super(OpenAIGPTModel, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.tokens_embed = nn.Embedding(config.input_dim, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
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
            self.layers[layer].attn.prune_heads(heads)

    def tie_weights(self):
        """
        Make sure we are sharing the input and output embeddings. Export to
        TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.pre_encoding, self.post_decoding)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor = None,
        inputs_raw: torch.FloatTensor = None,
        head_mask: torch.LongTensor = None,
    ) -> Tuple[Any, ...]:

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
        for i, block in enumerate(self.layers):
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


def recursive_product(x: torch.FloatTensor, dim: int, depth: int) -> torch.FloatTensor:
    """ Compute recursive unrolled product of (1 - x) along dim. """

    product_list: List[torch.FloatTensor] = []

    # TODO: Add a tensor of ones first.
    bsz, seq_len = x.shape[:2]
    product_list.append(torch.ones((bsz, seq_len)).to(x.device))

    # TODO: is this range correct?
    for i in range(depth - 1):
        if i == 0:
            product_list.append(1 - x[..., i + 1])
        else:
            product_list.append(product_list[i - 1] * (1 - x[..., i + 1]))

    # TODO: Should this be in-place?
    # TODO: Does this create a memory leak?
    products_tensor = torch.stack(product_list, dim=dim)

    return products_tensor


class ConditionalGPSTModel(OpenAIGPTPreTrainedModel):
    """ Orderbook transformer model for computing conditional over levels. """

    def __init__(self, config: OpenAIGPTConfig) -> None:
        super(ConditionalGPSTModel, self).__init__(config)

        self.modes = config.modes
        self.depth = config.orderbook_depth
        self.classification_modes = ["bid_classification", "ask_classification"]
        self.delta_modes = [
            "bid_increase",
            "bid_decrease",
            "ask_increase",
            "ask_decrease",
        ]

        self.transformers: Dict[str, OpenAIGPTLMHeadModel] = nn.ModuleDict()

        for mode in config.modes:
            subconfig = copy.deepcopy(config)
            subconfig.mode = mode
            if mode in ["bid_increase", "bid_decrease"]:
                subconfig.head_output_dim = self.depth
            elif mode in ["ask_increase", "ask_decrease"]:
                subconfig.head_output_dim = (2 * self.depth + 1) * self.depth
            elif mode == "bid_classification":
                subconfig.head_output_dim = 3
            elif mode == "ask_classification":
                subconfig.head_output_dim = (2 * self.depth + 1) * 3
            else:
                print("Value of config param ``mode``: %s" % mode)
                raise ValueError("Config param ``mode`` is invalid.")
            self.transformers[mode] = OpenAIGPTLMHeadModel(subconfig)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        labels: torch.LongTensor,
        inputs_raw: torch.FloatTensor,
        head_mask: torch.FloatTensor = None,
    ):
        depth = self.depth
        bsz, seq_len = input_ids.shape[:2]
        transformer_outputs: Dict[str, Tuple[Any, ...]] = {}
        transformer_logits: Dict[str, torch.FloatTensor] = {}
        for mode in self.modes:
            # TODO: Should we be passing ``labels`` here?
            transformer_outputs[mode] = self.transformers[mode](
                input_ids,
                position_ids=position_ids,
                inputs_raw=inputs_raw,
                head_mask=head_mask,
            )
            transformer_logits[mode] = transformer_outputs[mode][0]

        # Bid increase/decrease outputs have shape:
        #   ``(bsz, seq_len, orderbook_depth)``.

        # Ask increase/decrease outputs have shape:
        #   ``(bsz, seq_len, depth_range, orderbook_depth)``.

        # Bid classification outputs have shape:
        #   ``(bsz, seq_len, 3)``.

        # Ask classification outputs have shape:
        #   ``(bsz, seq_len, depth_range, 3)``.

        g_map: Dict[str, torch.FloatTensor] = {}
        g_logit_map: Dict[str, torch.FloatTensor] = {}
        dim_map: Dict[str, int] = {"bid": 2, "ask": 3}
        delta_index_map: Dict[str, int] = {"increase": 1, "decrease": 2}

        # TODO: Add shape annotations.
        for side in ["bid", "ask"]:
            # Components added in order: class, increase, decrease.
            g_components: List[torch.FloatTensor] = []

            dim = dim_map[side]
            mode = side + "_classification"

            class_outputs = transformer_logits[mode]

            # Add no-change logits for when ``y_1 == 0``.
            g_components.append(class_outputs[0])

            for delta in ["increase", "decrease"]:
                mode = side + "_" + delta
                sigmoid_outputs = torch.sigmoid(transformer_logits[mode])
                product_outputs = recursive_product(sigmoid_outputs, dim, depth)
                h = class_outputs[:, :, delta_index_map[delta]]

                # DEBUG
                print("Product outputs shape:", product_outputs.shape)
                print("h shape:", h.shape)

                # Shape check.
                assert sigmoid_outputs.shape == (bsz, seq_len, depth)
                assert product_outputs.shape == (bsz, seq_len, depth)
                assert h.shape == (bsz, seq_len)

                # Tile ``h`` across depth dimension.
                h = h.unsqueeze(dim)
                h = h.expand(-1, -1, depth)

                g_component = sigmoid_outputs * product_outputs * h

                # Shape check.
                if side == "bid":
                    assert g_component.shape == (bsz, seq_len, depth)
                else:
                    assert g_component.shape == (bsz, seq_len, (2 * depth + 1), depth)

                g_components.append(g_component)
            g_components = [comp.unsqueeze(dim) for comp in g_components]
            g = torch.stack(g_components, dim=dim)

            # Add un-tiled conditional distribution to outputs.
            g_logit_map[side] = g

            # TODO: Are we tiling across the correct dimension?
            if side == "bid":
                g = g.unsqueeze(dim + 1).expand(-1, -1, -1, (2 * depth + 1))
            assert g.shape == (bsz, seq_len, (2 * depth + 1), (2 * depth + 1))

            g_map[side] = g

        # TODO: Do we ever take log of the probabilities?
        q_logits = g_map["bid"] + g_map["ask"]

        # Shape check.
        assert q_logits.shape == (bsz, seq_len, (2 * depth + 1), (2 * depth + 1))
        logits = q_logits.reshape(bsz, seq_len, (2 * depth + 1) ** 2)

        # Construct ``outputs`` with logit map and subtransformer outputs.
        outputs = (g_logit_map,)
        for mode in self.modes:
            outputs += (transformer_outputs[mode],)

        # Shape of ``labels``: ``(bsz, seq_len)``.
        # Type of ``labels``: ``torch.LongTensor`` (integer values).
        # Range of values of ``labels``:
        #   ``0 <= label <= (2 * depth + 1)^2 - 1``.
        if labels is not None:
            # Shift so that tokens < n predict n.
            assert labels.shape == (bsz, seq_len)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens.
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss_input = shift_logits.view(-1, shift_logits.size(-1))
            loss_target = shift_labels.view(-1)
            loss = loss_fct(loss_input, loss_target)
            outputs = (loss,) + outputs

        return outputs


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

    def __init__(self, config: OpenAIGPTConfig) -> None:
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.depth_range = 2 * config.orderbook_depth + 1
        self.orderbook_depth = config.orderbook_depth
        self.mode = config.mode
        self.head = nn.Linear(config.input_dim, config.head_output_dim, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_raw: torch.FloatTensor,
        head_mask: torch.FloatTensor = None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,
            inputs_raw=inputs_raw,
            head_mask=head_mask,
        )
        hidden_states = transformer_outputs[0]
        assert hidden_states.shape == inputs_raw.shape

        logits = self.head(hidden_states)

        outputs = (logits,) + transformer_outputs[1:]

        return outputs  # last hidden state, (all hidden states), (all attentions)
