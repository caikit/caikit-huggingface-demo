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
import gradio as gr
import grpc


class TextGeneration:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, text_in):
        # False-y string doesn't work as required request param so '' --> ''
        return (
            self.predict(
                self.request(text_in=text_in), metadata=[("mm-model-id", model)]
            ).text
            if text_in
            else ""
        )

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
                inputs = gr.Textbox(
                    label='Input Text', placeholder=f'Prompt Text (hit enter to send)'
                )
                outputs = gr.Textbox(
                    label="Output Text", placeholder=f"Output text for {tab}"
                )
                inputs.submit(this.fn, [model_choice, inputs], outputs, api_name=tab)
                clear = gr.Button("Clear")

                def nones(*args):
                    return [None for x in args]
                clear.click(nones, [inputs, outputs], [inputs, outputs], queue=False)
                model_choice.change(
                    this.fn, [model_choice, inputs], outputs, api_name=tab
                )
                print(f"✅️  {tab} tab is enabled!")
                return True
        except grpc.RpcError as rpc_error:
            print(f"⚠️  Disabling {tab} tab due to:  {rpc_error.details()}")
            return False
