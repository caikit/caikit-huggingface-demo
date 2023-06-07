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
from module_ids import IMAGE_CLASSIFICATION
from runtime.data_model.classification import ClassificationPrediction, ClassInfo
from runtime.hf_base import HFBase
from transformers import pipeline

# Local
from caikit.core import ModuleBase, TaskBase, module, task

TASK = "image-classification"
# DEFAULTS: google/vit-base-patch16-224 and revision 5dca96d
# facebook/convnext-tiny-224


@task(
    required_parameters={"encoded_bytes_or_url": str},
    output_type=ClassificationPrediction,
)
class ImageClassificationTask(TaskBase):
    pass


@module(IMAGE_CLASSIFICATION, "image_classification", "0.0.0", ImageClassificationTask)
class ImageClassification(HFBase, ModuleBase):
    """Class to wrap image classification pipeline from Hugging Face"""

    def __init__(self, model_config_path) -> None:
        super().__init__()
        hf_model, hf_revision = self.read_config(model_config_path, None, None)
        self.pipe = pipeline(task=TASK, model=hf_model, revision=hf_revision)

    def run(
        self, encoded_bytes_or_url: str
    ) -> ClassificationPrediction:  # pylint: disable=arguments-differ
        """Run HF sentiment analysis
        Args:
            encoded_bytes_or_url Encoded image bytes (or url string)
        Returns:
            ClassificationPrediction: predicted classes with their confidence score.
        """

        image = HFBase.get_image_bytes(encoded_bytes_or_url)
        raw_results = self.pipe(image)  # , top_k=9)
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
