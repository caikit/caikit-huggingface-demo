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
import os

# Third Party
from module_ids import SENTIMENT
from runtime.data_model.classification import ClassificationPrediction, ClassInfo
from transformers import pipeline

# Local
from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, module


@module(SENTIMENT, "sentiment-analysis", "0.0.0")
class Sentiment(ModuleBase):
    """Class to wrap sentiment analysis pipeline from Hugging Face"""

    def __init__(self, model_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_path)
        config = loader.config
        model = pipeline(
            model=config.hf_model,
            revision=config.hf_revision,
            task="sentiment-analysis",
            return_all_scores=True,
        )
        self.sentiment_pipeline = model

    def run(
        self, text_in: str, **kwargs
    ) -> ClassificationPrediction:  # pylint: disable=arguments-differ
        """Run HF sentiment analysis
        Args:
            text_in str
        Returns:
            ClassificationPrediction: predicted classes with their confidence score.
        """
        raw_results = self.sentiment_pipeline([text_in])  # , top_k=9)
        class_info = []
        for result in raw_results:
            for sentiments in result:
                class_info.append(
                    ClassInfo(
                        class_name=sentiments["label"], confidence=sentiments["score"]
                    )
                )
        return ClassificationPrediction(class_info)

    @classmethod
    def bootstrap(cls, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
        """Bootstrap a module by loading a Hugging Face model."""
        return cls(model_path)

    @classmethod
    def load(cls, model_path):
        """Load a model."""
        return cls(model_path)
