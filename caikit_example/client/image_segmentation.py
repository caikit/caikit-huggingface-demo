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
from PIL import Image, ImageColor, ImageFont
import gradio as gr

# Color map for bbox color picking
COLORS = list(ImageColor.colormap.keys())
NUM_COLORS = len(COLORS)
FONT = ImageFont.truetype("Verdana.ttf", 20)


class ImageSegmentation:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, image_array):
        if image_array is None:
            return {}, None

        with Image.fromarray(image_array) as image:
            image_as_bytes = BytesIO()
            image.save(image_as_bytes, "PNG")  # Save into PNG file-like object
            encoded = base64.b64encode(
                image_as_bytes.getvalue()
            )  # Encode for transport

            response = self.predict(
                self.request(encoded_bytes_or_url=encoded),
                metadata=[("mm-model-id", model)],
            )

            results = [x for x in response.objects]
            results.sort(
                key=lambda c: c.score, reverse=True
            )  # Sort so numbering will be in desc score order

            labels = {}
            counter = {}
            gallery = []
            for result in results:
                label = result.label
                counter[label] = (
                    counter.get(label, 0) + 1
                )  # Count for repeated objects of same class
                key = (
                    label if counter[label] == 1 else f"{label}-{counter[label]}"
                )  # Append counter when repeated
                labels[key] = result.score

                with Image.open(BytesIO(base64.b64decode(result.mask))) as mask:
                    mask = mask.convert("L")
                    masked = image.copy()
                    masked.putalpha(mask)
                gallery.append((masked, key))

        return labels, gallery

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
            with gr.Row():
                with gr.Column():
                    inputs = gr.Image(label="Input Image")
                    output_image = gr.Gallery(
                        label="Output Masks", show_label=True
                    ).style(preview=True, object_fit="scale-down")
                with gr.Column():
                    outputs = gr.Label(label="Segments Detected")
                inputs.change(
                    this.fn,
                    [model_choice, inputs],
                    [outputs, output_image],
                    api_name=tab,
                )
            model_choice.change(
                this.fn, [model_choice, inputs], [outputs, output_image]
            )
            print(f"✅️  {tab} tab is enabled!")
            return True
