# Copyright The Caikit Authors
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
# limitations under the License

# Third Party
from caikit_library.block_ids import CONVERSATIONAL
from caikit_library.data_model.results import Text
from caikit_library.hf_base import HFBase
from transformers import Conversation, pipeline

# Local
from caikit.core import ModuleLoader, block

TASK = "conversational"

# Default model for task was microsoft/DialoGPT-medium (>800MB). Switched to -small (>350MB)
DEFAULT_MODEL = "microsoft/DialoGPT-small"
# Hard-coded revision to prevent download every revision bump.
DEFAULT_MODEL_REVISION = "4e936e3a11f8e077b31eec8f045499c92c7cf087"


@block(CONVERSATIONAL, TASK, "0.0.0")
class Conversational(HFBase):
    def __init__(self, model_config_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_config_path)
        hf_model, hf_revision = self.read_config(
            model_config_path, DEFAULT_MODEL, DEFAULT_MODEL_REVISION
        )
        self.pipe = pipeline(task=TASK, model=hf_model, revision=hf_revision)

    def run(self, text_in: str) -> Text:
        conversation = Conversation()
        conversation.add_user_input(text_in)
        conversation = self.pipe(conversation)
        return Text(conversation.generated_responses[-1])

    @classmethod
    def load(cls, model_path):
        return cls(model_path)
