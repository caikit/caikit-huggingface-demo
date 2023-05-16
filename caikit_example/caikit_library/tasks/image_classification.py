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
from caikit_library.block_ids import IMAGE_CLASSIFICATION
from caikit_library.data_model.classification import ClassificationPrediction, ClassInfo
from caikit_library.hf_base import HFBase
from PIL import Image
from transformers import pipeline
import requests

# Local
from caikit.core import ModuleLoader, ModuleSaver, block

TASK = "image-classification"
# DEFAULTS: google/vit-base-patch16-224 and revision 5dca96d
# facebook/convnext-tiny-224


@block(IMAGE_CLASSIFICATION, "image_classification", "0.0.0")
class ImageClassification(HFBase):
    """Class to wrap image classification  pipeline from huggingface"""

    def __init__(self, model_config_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_config_path)
        hf_model, hf_revision = self.read_config(model_config_path, None, None)
        self.pipe = pipeline(task=TASK, model=hf_model, revision=hf_revision)

    def run(self, url_in: str) -> ClassificationPrediction:
        """Run HF sentiment analysis
        Args:
            url_in Image in bytes
        Returns:
            ClassificationPrediction: predicted classes with their confidence score.
        """

        if os.path.isfile(url_in):
            image = Image.open(url_in)
        else:
            image = Image.open(requests.get(url_in, stream=True).raw)

        raw_results = self.pipe(image)  # , top_k=9)
        print("RAW RESULTS: ", raw_results)
        class_info = []
        for result in raw_results:
            class_info.append(
                ClassInfo(class_name=result["label"], confidence=result["score"])
            )
        return ClassificationPrediction(class_info)

    @classmethod
    def load(cls, model_config_path):
        """Load a model"""
        return cls(model_config_path)
