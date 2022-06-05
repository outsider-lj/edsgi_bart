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
__version__ = "3.0.2"
from modeling.file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_bart import BartConfig
from .tokenization_bart import BartTokenizer
from modeling.optimization import AdamW
from modeling.optimization import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from .modeling_edsgi_bart import (
    PretrainedBartModel,
    StepBartModel,
    StepBartForDialogueGeneration,
    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    shift_tokens_right,
    StepFocusedModel,
)