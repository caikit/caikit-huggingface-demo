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
from typing import List, Tuple

# Local
from caikit.core import DataObjectBase
from caikit.core.data_model import dataobject


@dataobject
class EmbeddingsPair(DataObjectBase):
    """pair an input token int with a list of output embedding floats"""

    input: int
    output: List[float]


@dataobject
class Result(DataObjectBase):
    """The result list of embeddings pairs"""

    output: List[EmbeddingsPair]
