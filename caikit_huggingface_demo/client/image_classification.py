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
from io import BytesIO
import base64

# Third Party
from PIL import Image
import gradio as gr


class ImageClassification:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, image_array):
        if image_array is None:
            return {}

        image_as_bytes = BytesIO()
        with Image.fromarray(image_array) as image:
            image.save(image_as_bytes, "PNG")  # Save into PNG file-like object

        encoded = base64.b64encode(image_as_bytes.getvalue())  # Encode for transport

        response = self.predict(
            self.request(encoded_bytes_or_url=encoded),
            metadata=[("mm-model-id", model)],
        )

        return {x.class_name: x.confidence for x in response.classes}

    @classmethod
    def optional_tab(cls, models, request, predict):
        if not models:
            return False

        tab = cls.__name__  # tab name
        this = cls(request, predict)
        with gr.Tab(tab):
            model_choice = gr.Dropdown(
                label="Model ID", choices=models, value=models[0]
            )
            inputs = gr.Image()
            outputs = gr.Label(label=tab, num_top_classes=9)
            inputs.change(this.fn, [model_choice, inputs], outputs, api_name=tab)
            model_choice.change(this.fn, [model_choice, inputs], outputs, api_name=tab)
            print(f"✅️  {tab} tab is enabled!")
            return True
