# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

__version__ = "4.1.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

# Configuration
from .configuration_utils import PretrainedConfig



# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)

# Tokenization
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)


# Integrations: this needs to come before other ml imports
# in order to allow any 3rd-party code to initialize properly
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Modeling
if is_torch_available():
    from .generation_beam_search import BeamScorer, BeamSearchScorer
    from .generation_logits_process import (
        HammingDiversityLogitsProcessor,
        LogitsProcessor,
        LogitsProcessorList,
        LogitsWarper,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PrefixConstrainedLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from .generation_utils import top_k_top_p_filtering
    from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer

    # Optimization
    from .optimization import (
        Adafactor,
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

    from .encoder_decoder import (
        EncoderDecoderModel,
        EncoderDecoderConfig,
    )
    from .bert import (
        BertModel,
        BertConfig,
        BertTokenizer,
        BertForSequenceClassification,
        BertLMHeadModel,
    )
    from .roberta import (
        RobertaModel,
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer
    )
    from .auto import (
        AutoConfig, AutoModel, AutoTokenizer
    )
    from .gpt2 import (
        GPT2Tokenizer,
        GPT2Model,
        GPT2Config,
    )


