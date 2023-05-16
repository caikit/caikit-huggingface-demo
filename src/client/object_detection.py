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
from PIL import Image, ImageColor, ImageDraw, ImageFont
import gradio as gr

# Color map for bbox color picking
COLORS = list(ImageColor.colormap.keys())
NUM_COLORS = len(COLORS)
FONT = ImageFont.truetype("Verdana.ttf", 20)


class ObjectDetection:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, image_array):
        if image_array is None:
            return {}, None

        image = Image.fromarray(image_array)
        image_as_bytes = BytesIO()
        image.save(image_as_bytes, "PNG")  # Save into PNG file-like object
        encoded = base64.b64encode(image_as_bytes.getvalue())  # Encode for transport

        response = self.predict(
            self.request(encoded_bytes_or_url=encoded),
            metadata=[("mm-model-id", model)],
        )

        results = list(response.objects)
        results.sort(
            key=lambda c: c.score, reverse=True
        )  # Sort so numbering will be in desc score order

        image_draw = ImageDraw.Draw(image)

        labels = {}
        counter = {}
        for result in results:
            label = result.label
            color = COLORS[
                hash(label) % NUM_COLORS
            ]  # Pick a color based on class name hash
            counter[label] = (
                counter.get(label, 0) + 1
            )  # Count for repeated objects of same class
            key = (
                label if counter[label] == 1 else f"{label}-{counter[label]}"
            )  # Append counter when repeated
            b = result.box
            text_size = FONT.getsize(key)
            label_size = (text_size[0] + 10, text_size[1] + 10)
            label_rectangle = Image.new(
                "RGBA", label_size, ImageColor.getrgb(color) + (0,)
            )  # with transparency
            label_draw = ImageDraw.Draw(label_rectangle)
            # label_draw.rectangle((5, 2)+label_size, fill=ImageColor.getrgb(color)+(127,))  # (255, 255, 0, 100))
            label_draw.text(
                (5, 2), key, font=FONT, fill="white"
            )  #  fill=(255, 255, 0, 255))

            image.paste(label_rectangle, (b.xmin, b.ymin))
            image_draw.rectangle(
                (b.xmin, b.ymin, b.xmax, b.ymax), outline=color, width=5
            )
            labels[key] = result.score

        return labels, image

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
                    output_image = gr.Image(label="Output Image")
                with gr.Column():
                    outputs = gr.Label(label="Objects Detected")  # , num_top_classes=9)
            inputs.change(
                this.fn, [model_choice, inputs], [outputs, output_image], api_name=tab
            )
            model_choice.change(
                this.fn, [model_choice, inputs], [outputs, output_image]
            )
            print(f"✅️  {tab} tab is enabled!")
            return True
