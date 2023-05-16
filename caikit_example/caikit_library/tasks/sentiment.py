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
from caikit_library.block_ids import SENTIMENT
from caikit_library.data_model.classification import ClassificationPrediction, ClassInfo
from transformers import pipeline

# Local
from caikit.core import BlockBase, ModuleLoader, ModuleSaver, block


@block(SENTIMENT, "sentiment-analysis", "0.0.0")
class Sentiment(BlockBase):
    """Class to wrap sentiment analysis pipeline from huggingface"""

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

    def run(self, text_in: str) -> ClassificationPrediction:
        """Run HF sentiment analysis
        Args:
            text_in TextInput
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
        """Load a huggingface based watson-core model
        Args:
            model_path: str
                Path to hugging-face model
        Returns:
            HuggingFaceExampleModel
        """
        return cls(model_path)

    def save(self, model_path, **kwargs):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
            library_name="hf_example",
            library_version="1.2.3",
        )

        # Extract object to be saved
        with module_saver:
            # Make the directory to save model artifacts
            rel_path, _ = module_saver.add_dir("hf_model")
            save_path = os.path.join(model_path, rel_path)
            self.sentiment_pipeline.save_pretrained(save_path)
            module_saver.update_config({"hf_artifact_path": rel_path})

    # this is how you load the model, if you have a watson core model
    @classmethod
    def load(cls, model_path):
        """Load a huggingface based watson-core model
        Args:
            model_path: str
                Path to hugging-face model
        Returns:
            HuggingFaceExampleModel
        """
        return cls(model_path)
