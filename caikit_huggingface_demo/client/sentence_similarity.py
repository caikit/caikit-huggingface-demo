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
from sentence_transformers import util
import gradio as gr
import numpy
import pandas


class SentenceSimilarity:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, sentence_0, sentence_1, sentence_2):
        sentences = [sentence_0, sentence_1, sentence_2]
        response = self.predict(
            self.request(sentences=sentences), metadata=[("mm-model-id", model)]
        )
        columns = []
        rows = []
        for k in response.output:
            if k.input == 0:
                columns.append(f"Source sentence: {sentences[k.input]}")
            else:
                columns.append(f"Sentence {k.input}: {sentences[k.input]}")

        for r, _ in enumerate(response.output[0].output):
            row = []
            for c, _ in enumerate(columns):
                row.append(response.output[c].output[r])
            rows.append(row)

        embedding_0 = response.output[0].output
        output_cos = {
            columns[c]: util.cos_sim(embedding_0, response.output[c].output).item()
            for c, v in enumerate(columns)
        }
        ret_embeddings = pandas.DataFrame(numpy.array(rows), columns=columns)
        return output_cos, ret_embeddings

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
            input_0 = gr.Textbox(
                label="Source Sentence",
                placeholder=f"Enter input text for source sentence",
            )
            input_1 = gr.Textbox(
                label="Sentence 1",
                placeholder=f"Enter input text for sentence to score",
            )
            input_2 = gr.Textbox(
                label="Sentence 2",
                placeholder=f"Enter input text for another sentence to score",
            )
            clear = gr.Button("Submit")
            output_cos = gr.Label(label="Cosine Similarity")
            output_embeddings = gr.Dataframe()
            outputs = [output_cos, output_embeddings]

            clear.click(
                this.fn,
                [model_choice, input_0, input_1, input_2],
                outputs,
                api_name=tab,
            )
            model_choice.change(
                this.fn,
                [model_choice, input_0, input_1, input_2],
                outputs,
                api_name=tab,
            )

            print(f"✅️  {tab} tab is enabled!")
            return True
