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
from PIL import Image
import gradio as gr
import grpc


class ImageClassification:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, img):
        if img is None:
            return {}

        # image_as_bytes = bytes(Image.fromarray(img).tobytes())
        response = self.predict(
            self.request(url_in=img), metadata=[("mm-model-id", model)]
        )
        # response = self.predict(self.request(image_bytes=image_as_bytes), metadata=[("mm-model-id", model)])
        return {x.class_name: x.confidence for x in response.classes}

    @classmethod
    def optional_tab(cls, models, request, predict):
        if not models:
            return False

        tab = cls.__name__  # tab name
        try:
            this = cls(request, predict)
            with gr.Tab(tab):
                model_choice = gr.Dropdown(
                    label="Model ID", choices=models, value=models[0]
                )
                inputs = gr.Textbox()
                # inputs = gr.Image()
                outputs = gr.Label(label=tab, num_top_classes=9)
                inputs.change(this.fn, [model_choice, inputs], outputs, api_name=tab)
                model_choice.change(
                    this.fn, [model_choice, inputs], outputs, api_name=tab
                )
                print(f"✅️  {tab} tab is enabled!")
                return True
        except grpc.RpcError as rpc_error:
            print(f"⚠️  Disabling {tab} tab due to:  {rpc_error.details()}")
            return False
