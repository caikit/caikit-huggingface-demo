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
from typing import List

# Local
from caikit.core import DataObjectBase
from caikit.core.data_model import dataobject


@dataobject
class Mask(DataObjectBase):
    label: str  # (required) Predicted relevant class name
    score: float  # (required) The confidence-like score of this prediction in [0, 1]
    mask: str  # (required) Mask for the layer encoded into a str


@dataobject
class ImageSegmentationResult(DataObjectBase):
    """The result of image-segmentation inference."""

    objects: List[Mask]
