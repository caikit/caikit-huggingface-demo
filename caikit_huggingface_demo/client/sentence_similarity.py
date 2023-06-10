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


class SentenceSimilarity:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, sentence_0, sentence_1, sentence_2):
        # if not sentence_0:  # False-y string doesn't work as required request param
            # return ""

        response = self.predict(
            self.request(sentences=[sentence_0, sentence_1, sentence_2]), metadata=[("mm-model-id", model)]
        )
        columns = []
        rows = []
        for k in response.output:
            columns.append(k.input)

        for r, _ in enumerate(response.output[0].output):
            row = []
            for c, _ in enumerate(columns):
                row.append(response.output[c].output[r])
            rows.append(row)

        return pandas.DataFrame(numpy.array(rows), columns=columns)

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
                input_0 = gr.Textbox(
                    label="Sentence 0", placeholder=f"Enter input text for {tab} "
                )
                input_1 = gr.Textbox(
                    label="Sentence 1", placeholder=f"Enter input text for {tab}"
                )
                input_2 = gr.Textbox(
                    label="Sentence 2", placeholder=f"Enter input text for {tab}"
                )
                clear = gr.Button("Submit")
                outputs = gr.Dataframe()

                clear.click(
                    this.fn, [model_choice, input_0, input_1, input_2], outputs, api_name=tab
                )
                model_choice.change(
                    this.fn, [model_choice, input_0, input_1, input_2], outputs, api_name=tab
                )

                print(f"✅️  {tab} tab is enabled!")
                return True
        except grpc.RpcError as rpc_error:
            print(f"⚠️  Disabling {tab} tab due to:  {rpc_error.details()}")
            return False
