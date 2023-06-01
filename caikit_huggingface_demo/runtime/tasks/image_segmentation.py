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
from block_ids import IMAGE_SEGMENTATION
from runtime.data_model.image_segmentation import ImageSegmentationResult, Mask
from runtime.hf_base import HFBase
from transformers import pipeline

# Local
from caikit.core import ModuleBase, module

PIPE_TASK = "image-segmentation"
TASK_NAME = PIPE_TASK.replace("-", "_")

# HF Default is:
# - facebook/detr-resnet-50-panoptic and revision fc15262
# - size 172MB
# - requires timm
# Pinning that revision as default
DEFAULT_HF_MODEL = "facebook/detr-resnet-50-panoptic"
DEFAULT_HF_MODEL_REVISION = "fc15262"


@module(IMAGE_SEGMENTATION, TASK_NAME, "0.0.0")
class ImageSegmentation(HFBase, ModuleBase):
    """Class to wrap image-segmentation pipeline from Hugging Face"""

    def __init__(self, model_config_path) -> None:
        super().__init__()
        hf_model, hf_revision = self.read_config(
            model_config_path, DEFAULT_HF_MODEL, DEFAULT_HF_MODEL_REVISION
        )
        self.pipe = pipeline(task=PIPE_TASK, model=hf_model, revision=hf_revision)

    def run(
        self, encoded_bytes_or_url: str
    ) -> ImageSegmentationResult:  # pylint: disable=arguments-differ
        image = HFBase.get_image_bytes(encoded_bytes_or_url)
        results = self.pipe(image, threshold=0.5)
        objects = [
            Mask(
                label=o["label"], score=o["score"], mask=HFBase.encode_image(o["mask"])
            )
            for o in results
        ]
        return ImageSegmentationResult(objects)

    @classmethod
    def load(cls, model_config_path):
        """Load a model given a Caikit model config dir"""
        return cls(model_config_path)
