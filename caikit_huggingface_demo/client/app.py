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
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import MessageFactory
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)
import gradio as gr
import grpc
import module_ids

# Local
from .conversational import Conversational
from .embeddings import Embeddings
from .image_classification import ImageClassification
from .image_segmentation import ImageSegmentation
from .object_detection import ObjectDetection
from .sentence_similarity import SentenceSimilarity
from .sentiment import Sentiment
from .summarization import Summarization
from .text_generation import TextGeneration
from caikit.runtime.service_factory import ServicePackage


def add_tab(ui_class, client_stub, service_prefix, desc_pool, module_models):
    """Adds a tab if there are models loaded for this module.
    returns true if tab added else false (no models)
    """
    class_name = ui_class.__name__
    task = f"{class_name}Task"
    method_name = f"{task}Predict"
    if hasattr(client_stub, method_name):
        method = getattr(client_stub, method_name)
    else:
        print(f"Failed to find expected method: {method_name}")
        return False

    request_name = f"{service_prefix}.{task}Request"
    try:
        request_desc = desc_pool.FindMessageTypeByName(request_name)
    except KeyError as e:
        print(f"Find request error: {e}")
        return False

    request = MessageFactory(desc_pool).GetPrototype(request_desc)
    models = module_models.get(module_ids.MODULE_IDS[class_name])
    return ui_class.optional_tab(models, request, method)


def get_frontend(
    channel: grpc.Channel, inference_service: ServicePackage, module_models: dict
) -> gr.Blocks:
    client_stub = inference_service.stub_class(channel)
    reflection_db = ProtoReflectionDescriptorDatabase(channel)
    desc_pool = DescriptorPool(reflection_db)
    services = [
        x for x in reflection_db.get_services() if x.startswith("caikit.runtime.") and not (
            x.startswith("caikit.runtime.training") or x.startswith("caikit.runtime.training"))]
    if len(services) != 1:
        print(f"Error: Expected 1 caikit.runtime service, but found {len(services)}.")
    service_name = services[0]
    service_prefix, _, _ = service_name.rpartition(".")

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
        for ui_class in [
            Conversational,
            TextGeneration,
            Summarization,
            Sentiment,
            SentenceSimilarity,
            Embeddings,
            ImageClassification,
            ObjectDetection,
            ImageSegmentation,
        ]:
            tabs |= add_tab(
                ui_class, client_stub, service_prefix, desc_pool, module_models
            )

        if not tabs:
            print("!!! NO UI TABS WERE SUCCESSFULLY LOADED !!!")
            print("  ^ Is the gRPC server even running?")
            print(
                "  ^ Did you add model configurations under the caikit_huggingface_demo/models directory?"
            )

    return frontend
