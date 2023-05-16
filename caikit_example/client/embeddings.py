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
import numpy
import pandas


class Embeddings:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, text_in):
        if not text_in:  # False-y string doesn't work as required request param
            return pandas.DataFrame()

        response = self.predict(
            self.request(text_in=text_in), metadata=[("mm-model-id", model)]
        )
        columns = []
        rows = []
        for k in response.output:
            columns.append(k.input)

        for r in range(len(response.output[0].output)):
            row = []
            for c in range(len(columns)):
                row.append(response.output[c].output[r])
            rows.append(row)

        ret = pandas.DataFrame(numpy.array(rows), columns=columns)
        return ret

    @classmethod
    def optional_tab(cls, models, request, predict):
        if not models:
            return False

        tab = cls.__name__  # tab name
        this = cls(request, predict)
        with gr.Tab("Embeddings"):
            model_choice = gr.Dropdown(
                label="Model ID", choices=models, value=models[0]
            )
            inputs = gr.Textbox(
                label="Input Text", placeholder=f"Enter input text for {tab}"
            )
            outputs = gr.Dataframe()
            inputs.change(this.fn, [model_choice, inputs], outputs, api_name=tab)
            model_choice.change(this.fn, [model_choice, inputs], outputs, api_name=tab)
            print(f"✅️  {tab} tab is enabled!")
            return True
