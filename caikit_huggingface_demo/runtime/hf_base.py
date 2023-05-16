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
from base64 import b64decode, b64encode
from io import BytesIO
import os

# Third Party
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests

# Local
from caikit.core import BlockBase, ModuleConfig
from caikit.core.blocks import BlockSaver

DEFAULT_MODEL = None
DEFAULT_MODEL_REVISION = None


class HFBase(BlockBase):
    def __init__(self, model=None, tokenizer=None) -> None:
        """This function gets called by `.load` and `.train` function
        which initializes this module.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def save(self, artifact_path, *args, **kwargs):
        block_saver = BlockSaver(self, model_path=artifact_path)

        # Extract object to be saved
        with block_saver:
            # TODO: didn't save to models sub-dir.  Also did not re-test w/ this change too.
            # block_saver.update_config({"artifact": "models"})
            block_saver.update_config({"artifact": "."})
            if self.tokenizer:  # This condition allows for empty placeholders
                self.tokenizer.save_pretrained(artifact_path)
            if self.model:  # This condition allows for empty placeholders
                self.model.save_pretrained(artifact_path)

    @classmethod
    def read_config(cls, model_name_or_path, default_model, default_model_revision):
        config = ModuleConfig.load(model_name_or_path)
        model_name = config.get("hf_model", default_model)
        model_revision = config.get("hf_model_revision", default_model_revision)
        return model_name, model_revision

    @classmethod
    def load(cls, model_config_path: str):
        model_name, model_revision = cls.read_config(
            model_config_path, DEFAULT_MODEL, DEFAULT_MODEL_REVISION
        )
        return cls.bootstrap(model_name, revision=model_revision)

    @classmethod
    def bootstrap(cls, pretrained_model_name_or_path: str, revision=None):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, revision=revision
        )
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path, revision=revision
        )
        return cls(model, tokenizer)

    @classmethod
    def get_image_bytes(cls, encoded_bytes_or_url: str) -> Image:
        """Take input string (url, path-to-file, or encoded bytes) and return a PIL Image."""

        # Get image from URL
        if encoded_bytes_or_url.startswith("http"):
            return Image.open(
                requests.get(encoded_bytes_or_url, stream=True, timeout=60).raw
            )

        # Get image from local file. Handy for local demo/test.
        # Simple length limit check to avoid trying to use image data as a path.
        if len(encoded_bytes_or_url) < 256 and os.path.isfile(encoded_bytes_or_url):
            return Image.open(encoded_bytes_or_url)

        # Decode and open the image bytes
        return Image.open(BytesIO(b64decode(encoded_bytes_or_url)))

    @classmethod
    def encode_image(cls, image: Image) -> bytes:
        image_as_bytes = BytesIO()
        image.save(image_as_bytes, "PNG")  # Save into PNG file-like object
        return b64encode(image_as_bytes.getvalue())  # Encode for transport
