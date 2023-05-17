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
from caikit_library.block_ids import OBJECT_DETECTION
from caikit_library.data_model.object_detection import (
    Box,
    ObjectDetected,
    ObjectDetectionResult,
)
from caikit_library.hf_base import HFBase
from transformers import pipeline

# Local
from caikit.core import block

PIPE_TASK = "object-detection"
TASK_NAME = PIPE_TASK.replace("-", "_")

# HF pipeline default:
# - facebook/detr-resnet-50 and revision 2729413
#   - requires timm
#   - 167MB
# So we are using a tiny default model instead.
DEFAULT_HF_MODEL = "hustvl/yolos-tiny"
DEFAULT_HF_MODEL_REVISION = "3686e65df0c914833fc8cbeca745a33b374c499b"


@block(OBJECT_DETECTION, TASK_NAME, "0.0.0")
class ObjectDetection(HFBase):
    """Class to wrap object-detection pipeline from Hugging Face"""

    def __init__(self, model_config_path) -> None:
        super().__init__()
        hf_model, hf_revision = self.read_config(
            model_config_path, DEFAULT_HF_MODEL, DEFAULT_HF_MODEL_REVISION
        )
        self.pipe = pipeline(task=PIPE_TASK, model=hf_model, revision=hf_revision)

    def run(self, encoded_bytes_or_url: str) -> ObjectDetectionResult:
        image = HFBase.get_image_bytes(encoded_bytes_or_url)
        results = self.pipe(image, threshold=0.5)
        objects = [
            ObjectDetected(label=o["label"], score=o["score"], box=Box(**o["box"]))
            for o in results
        ]
        return ObjectDetectionResult(objects)

    @classmethod
    def load(cls, model_config_path):
        """Load a model given a Caikit model config dir"""
        return cls(model_config_path)
