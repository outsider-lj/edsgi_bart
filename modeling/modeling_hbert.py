# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


import logging
from typing import Optional
from models.bert.modeling_bert import BertModel, BertLayer,BertLMHeadModel,BertEmbeddings,BertOnlyMLMHead,BertPooler,BertPreTrainedModel
from models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from models.configuration_utils import PretrainedConfig
from models.modeling_utils import PreTrainedModel
from models.encoder_decoder import EncoderDecoderModel
import torch
import torch.nn as nn
import copy

logger = logging.getLogger(__name__)
class Integration(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.embeddings = BertEmbeddings(config)
        self.layer = BertLayer(config)
        # self.pooler = BertPooler(config) if add_pooling_layer else None
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
            decoder_steps_states=None,
            decoder_steps_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        outputs = self.layer(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 32)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HBertModel(PreTrainedModel):
    r"""
        :class:`~transformers.EncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method for the encoder and `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` class method for the decoder.
    """
    config_class = EncoderDecoderConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder1: Optional[PreTrainedModel] = None,
        decoder2: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder1 is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder1.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)
        # self.integration_config.num_hidden_layers=1
        if encoder is None:
            from models.auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder1 is None:
            from models.auto.modeling_auto import AutoModelForCausalLM

            decoder1 = AutoModelForCausalLM.from_config(config.decoder)
        if decoder2 is None:
            from models.auto.modeling_auto import AutoModelForCausalLM
            decoder2 = AutoModelForCausalLM.from_config(config.decoder)
        config.decoder.num_hidden_layers=12
        self.encoder = encoder
        self.decoder_emo = decoder1
        self.decoder_topic= decoder2
        integration_config=copy.copy(config.decoder)
        integration_config.num_hidden_layers=1
        self.integration=Integration(integration_config)#decoder
        # self.lm_head_emo=torch.nn.Linear(decoder.config.hidden_size,decoder.config.vocab_size)
        # self.lm_head_topic=torch.nn.Linear(decoder.config.hidden_size,decoder.config.vocab_size)
        # self.lm_head_final=torch.nn.Linear(decoder.config.hidden_size,decoder.config.vocab_size)
        self.emo_cls=ClassificationHead(encoder.config)
            # torch.nn.Linear(decoder.config.hidden_size,32)
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        emo_output_embeddings,topic_output_embeddings = self.get_output_embeddings()
        if emo_output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(emo_output_embeddings, self.get_input_embeddings())
        if topic_output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(emo_output_embeddings, self.get_input_embeddings())
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder_emo.get_output_embeddings(),self.decoder_topic.get_output_embeddings()

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.

            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`).

                - To update the encoder configuration, use the prefix `encoder_` for each configuration parameter.
                - To update the decoder configuration, use the prefix `decoder_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import EncoderDecoderModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")

        """

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder1 = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        kwargs_decoder2 = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder1.keys():
            del kwargs["decoder_" + key]
        for key in kwargs_decoder2.keys():
            del kwargs["decoder_" + key]
        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from models.auto.modeling_auto import AutoModel

            if "config" not in kwargs_encoder:
                from models.auto.configuration_auto import AutoConfig

                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder1 = kwargs_decoder1.pop("model", None)
        if decoder1 is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from models.auto.modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder1:
                from models.auto.configuration_auto import AutoConfig

                decoder_config1 = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config1.is_decoder is False or decoder_config1.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config1.is_decoder = True
                    decoder_config1.add_cross_attention = True

                kwargs_decoder1["config"] = decoder_config1

            if kwargs_decoder1["config"].is_decoder is False or kwargs_decoder1["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder1 = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder1)
        decoder2 = kwargs_decoder2.pop("model", None)
        if decoder2 is None:
            assert (
                        decoder_pretrained_model_name_or_path is not None
                ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from models.auto.modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder2:
                from models.auto.configuration_auto import AutoConfig

                decoder_config2 = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config2.is_decoder is False or decoder_config1.add_cross_attention is False:
                    logger.info(
                            f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config2.is_decoder = True
                    decoder_config2.add_cross_attention = True

                kwargs_decoder2["config"] = decoder_config2

            if kwargs_decoder2["config"].is_decoder is False or kwargs_decoder2[
                    "config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder2 = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder2)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config1)
        return cls(encoder=encoder, decoder1=decoder1,decoder2=decoder2,config=config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_mask_matrix=None,
            encoder_token_type_ids=None,
            decoder_input_ids_first=None,
            decoder_input_ids_second=None,
            decoder_input_ids_final=None,
            decoder_attention_mask_first=None,
            decoder_attention_mask_second=None,
            decoder_attention_mask_final=None,
            encoder_outputs=None,
            past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,  # TODO: (PVP) implement :obj:`use_cache`
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_train=True,
            is_integration=True,
            **kwargs,
    ):

        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the encoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
                Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
                `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder.
            decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
                Provide for sequence to sequence training to the decoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
                Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            decoder_head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the decoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the left-to-right language modeling loss (next word prediction) for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:
                - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
                - With a `decoder_` prefix which will be input as `**decoder_kwargs` for the decoder forward function.

        Examples::

            from transformers import EncoderDecoderModel, BertTokenizer
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert

            # forward
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            # training
            loss, outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)[:2]

            # generation
            generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        kwargs_decoder = {}

         # Decode
        decoder_outputs_emo =  self.decoder_emo(
            input_ids=decoder_input_ids_first,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            attention_mask=decoder_attention_mask_first,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            # inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs_decoder,
        )

        decoder_outputs_topic = self.decoder_topic(
            input_ids=decoder_input_ids_second,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            attention_mask=decoder_attention_mask_second,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            # inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs_decoder,
        )
        emo_states=decoder_outputs_emo.hidden_states.detach()
        topic_states=decoder_outputs_topic.hidden_states.detach()
        decoder_steps_states=torch.cat((emo_states,topic_states),1)
        decoder_steps_mask=torch.cat((decoder_attention_mask_first,decoder_attention_mask_second),1)
        # decoder_inputs_embeds=self.integration.base_model.embeddings(decoder_input_ids_final)
        # decoder_inputs_embeds=torch.cat((decoder_steps_states,decoder_inputs_embeds),dim=1)
        # decoder_attention_mask_final=torch.cat((decoder_steps_mask,decoder_attention_mask_final),dim=1)
        decoder_outputs_final=self.integration(
            input_ids=decoder_input_ids_final,
            attention_mask=decoder_attention_mask_final,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_steps_states=decoder_steps_states,
            decoder_steps_mask=decoder_steps_mask,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        # decoder_outputs_final = self.integration(
        #     input_ids=None,
        #     attention_mask=decoder_steps_mask,
        #     inputs_embeds=decoder_steps_states,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     **kwargs_encoder,
        # )
        output_emo=decoder_outputs_emo.logits#.detach()
        output_topic=decoder_outputs_topic.logits#.detach()
        # output_emo = self.lm_head_emo(emo_states.detach())
        # output_topic=self.lm_head_topic(topic_states.detach())
        # output_final = self.lm_head_final(decoder_outputs_final[0])
        emo_logits=self.emo_cls(encoder_hidden_states)
        return output_emo,output_topic,decoder_outputs_final,emo_logits

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, encoder_outputs=None, **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

        # Ideally all models should have a :obj:`use_cache`
        # leave following to ifs until all have it implemented
        if "use_cache" in decoder_inputs:
            input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

        if "past_key_values" in decoder_inputs:
            input_dict["past_key_values"] = decoder_inputs["past_key_values"]

        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)
