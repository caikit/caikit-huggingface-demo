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
from caikit_library.block_ids import *
from client.conversational import Conversational
from client.embeddings import Embeddings
from client.image_classification import ImageClassification
from client.image_segmentation import ImageSegmentation
from client.object_detection import ObjectDetection
from client.sentiment import Sentiment
from client.summarization import Summarization
from client.text_generation import TextGeneration
import gradio as gr
import grpc

# Local
from caikit.runtime.service_factory import ServicePackage


def get_frontend(
    channel: grpc.Channel, inference_service: ServicePackage, module_models: dict
) -> gr.Blocks:
    client_stub = inference_service.stub_class(channel)

    # Build client UI with gradio
    with gr.Blocks(analytics_enabled=False) as frontend:
        gr.Markdown(
            """
            # Hugging Face examples on running on Caikit
            ## Each tab below represents:
               * A Hugging Face gradio UI component
               * Pretrained model(s) from Hugging Face loaded by Caikit
               * Output from a gRPC inference endpoint provided by Caikit
            
            ## Try it!
            You should see one tab below for each example UI component that found its loaded Caikit module+models. 
            Pick a tab, choose a model, enter some input, see the results, repeat.
            """
        )

        tabs = False
        tabs |= Conversational.optional_tab(
            module_models.get(CONVERSATIONAL),
            inference_service.messages.ConversationalRequest,
            client_stub.ConversationalPredict,
        )
        tabs |= TextGeneration.optional_tab(
            module_models.get(TEXT_GENERATION),
            inference_service.messages.TextGenerationRequest,
            client_stub.TextGenerationPredict,
        )
        tabs |= Summarization.optional_tab(
            module_models.get(SUMMARIZATION),
            inference_service.messages.SummarizationRequest,
            client_stub.SummarizationPredict,
        )
        tabs |= Sentiment.optional_tab(
            module_models.get(SENTIMENT),
            inference_service.messages.SentimentRequest,
            client_stub.SentimentPredict,
        )
        tabs |= Embeddings.optional_tab(
            module_models.get(EMBEDDINGS),
            inference_service.messages.EmbeddingsRequest,
            client_stub.EmbeddingsPredict,
        )
        tabs |= ImageClassification.optional_tab(
            module_models.get(IMAGE_CLASSIFICATION),
            inference_service.messages.ImageClassificationRequest,
            client_stub.ImageClassificationPredict,
        )
        tabs |= ObjectDetection.optional_tab(
            module_models.get(OBJECT_DETECTION),
            inference_service.messages.ObjectDetectionRequest,
            client_stub.ObjectDetectionPredict,
        )
        tabs |= ImageSegmentation.optional_tab(
            module_models.get(IMAGE_SEGMENTATION),
            inference_service.messages.ObjectDetectionRequest,
            client_stub.ImageSegmentationPredict,
        )

        if not tabs:
            print("!!! NO UI TABS WERE SUCCESSFULLY LOADED !!!")
            print(f"  ^ Is the gRPC server even running?")
            print(
                f"  ^ Did you add model configurations under the caikit_example/models directory?"
            )

    return frontend
