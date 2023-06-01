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
from module_ids import EMBEDDINGS
from runtime.data_model.embeddings import EmbeddingsPair, Result
from transformers import AutoModel, AutoTokenizer
import torch

# Local
from caikit.core import ModuleBase, ModuleConfig, module


@module(EMBEDDINGS, "embeddings", "0.0.0")
class Embeddings(ModuleBase):
    def __init__(self, tokenizer=None, model=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def run(self, text_in: str) -> Result:  # pylint: disable=arguments-differ
        model_input = self.tokenizer(text_in, return_tensors="pt")
        model_output = self.model(**model_input)
        x = torch.squeeze(model_output.last_hidden_state)
        with torch.no_grad():
            input_ids = torch.squeeze(model_input.input_ids).detach().cpu().numpy()

            embeddings_pairs = []
            for i, out in enumerate(x.detach().cpu().numpy()):
                outputs = [o.item() for o in out]
                embeddings_pairs.append(
                    EmbeddingsPair(input=input_ids[i], output=outputs)
                )

        return Result(embeddings_pairs)

    @classmethod
    def load(cls, model_path: str):
        # Read config file
        config = ModuleConfig.load(model_path)
        model_name = config.hf_model or "distilbert-base-uncased"
        model_revision = (
            config.hf_model_revision or "1c4513b2eedbda136f57676a34eea67aba266e5c"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        model = AutoModel.from_pretrained(model_name)
        return cls(tokenizer, model)

    @classmethod
    def bootstrap(cls, pretrained_model_name_or_path: str):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        return cls(tokenizer, model)
