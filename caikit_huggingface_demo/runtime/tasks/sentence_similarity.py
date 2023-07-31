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

# Standard
from pathlib import Path
from typing import List

# Third Party
from module_ids import SENTENCE_SIMILARITY
from runtime.data_model.embeddings import EmbeddingsPair, Result
from runtime.hf_base import HFBase
from sentence_transformers import SentenceTransformer

# Local
from caikit.core import ModuleBase, TaskBase, module, task

DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HOME = Path.home()


@task(
    required_parameters={"sentences": List[str]},
    output_type=Result,
)
class SentenceSimilarityTask(TaskBase):
    pass


@module(SENTENCE_SIMILARITY, "sentence-similarity", "0.0.0", SentenceSimilarityTask)
class SentenceSimilarity(HFBase, ModuleBase):
    """Class to wrap sentence-similarity models with sentence transformers"""

    def __init__(self, model_config_path) -> None:
        super().__init__()
        hf_model, _hf_revision = self.read_config(
            model_config_path, DEFAULT_HF_MODEL, None
        )
        self.model = SentenceTransformer(
            hf_model, cache_folder=f"{HOME}/.cache/huggingface/sentence_transformers"
        )

    def run(
        self, sentences: List[str], **kwargs
    ) -> Result:  # pylint: disable=arguments-differ
        embeddings = self.model.encode(sentences)

        results: List[EmbeddingsPair] = []
        for i, e in enumerate(embeddings):
            results.append(EmbeddingsPair(input=i, output=e))
        return Result(results)

    @classmethod
    def load(cls, model_path):
        """Load a model."""
        return cls(model_path)
