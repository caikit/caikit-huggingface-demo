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
from block_ids import SUMMARIZATION
from runtime.data_model.results import Text
from runtime.hf_base import HFBase
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Local
from caikit.core import ModuleBase, module

DEFAULT_MODEL = "JulesBelveze/t5-small-headline-generator"
DEFAULT_MODEL_REVISION = "0db30a2"


@module(id=SUMMARIZATION, name="summarization", version="0.0.0")
class Summarization(HFBase, ModuleBase):
    def run(self, text_in: str) -> Text:  # pylint: disable=arguments-differ
        input_ids = self.tokenizer(text_in, return_tensors="pt")["input_ids"]
        output_ids = self.model.generate(input_ids)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return Text(summary)

    @classmethod
    def load(cls, model_config_path: str):
        model_name, model_revision = HFBase.read_config(
            model_config_path, DEFAULT_MODEL, DEFAULT_MODEL_REVISION
        )

        # Instantiate from pretrained
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=model_revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        return cls(model, tokenizer)
