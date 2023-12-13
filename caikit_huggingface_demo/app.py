#!/usr/bin/env python
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
# limitations under the License.

# Standard
import argparse
import os
import sys

# Third Party
from client.app import get_frontend
import grpc

# Local
from caikit.config import configure, get_config
from caikit.core import ModuleConfig
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackageFactory

# runtime library config
CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "runtime", "config", "config.yml")
)
configure(CONFIG_PATH)


def _get_module_models(model_manager=None) -> dict:
    """
    Determine the modules and models that are loaded (so the UI only shows what is available).
    Tries to use model manager to keep backend/frontend in sync, but can run independently by
    using the same configs that the backend would use (library runtime config.yml and model config.yml files.

    Parameters
    ----------
    model_manager - if set this allows us to ask the ModelManager for loaded models (and details) instead of crawling configs and assuming the backend used the same configs.

    Returns
    -------
    dict - mapping loaded module module_ids to loaded model model_ids

    """
    if model_manager:
        model_modules = {
            k: v.model().MODULE_ID
            for (k, v) in model_manager.loaded_models.items()
        }
    else:
        model_modules = {}
        # Without loading models build a map from local_models_dir configs
        local_models_path = get_config().runtime.local_models_dir
        if os.path.exists(local_models_path):
            for model_id in os.listdir(local_models_path):
                try:
                    # Use the file name as the model id
                    model_path = os.path.join(local_models_path, model_id)
                    config = ModuleConfig.load(model_path)
                    model_modules[model_id] = config.module_id
                except Exception:  # pylint: disable=broad-exception-caught
                    # Broad exception, but want to ignore any unusable dirs/files
                    pass

    flipped = {}  # map module_id to list of model_ids
    for k, v in model_modules.items():
        flipped[v] = flipped.get(v, []) + [k]
    return flipped


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Starts app backend (gRPC) and/or frontend (gradio UI). Arguments "
        "allow running backend or frontend independently. The default "
        "is to run both in one process."
    )
    # Pattern emulates action=argparse.BooleanOptionalAction but that requires Python >= 3.9
    parser.add_argument(
        "--backend", default=None, action="store_true", help="Flag for backend-only"
    )
    parser.add_argument(
        "--no-backend",
        dest="backend",
        action="store_false",
        help="Flag to run without backend",
    )
    parser.add_argument(
        "--frontend", default=None, action="store_true", help="Flag for frontend-only"
    )
    parser.add_argument(
        "--no-frontend",
        dest="frontend",
        action="store_false",
        help="Flag to run without frontend",
    )
    args = parser.parse_args()
    # By default backend and frontend both start unless explicitly disabled with --no-backend or --no-frontend
    # or if one is explicitly enabled with --backend or --frontend and the other is not.
    backend = args.backend or args.backend is None and not args.frontend
    frontend = args.frontend or args.frontend is None and not args.backend

    if backend and frontend:
        print("Command-line enabled Caikit gRPC backend server and frontend gradio UI")
    else:
        print("Command-line disabled backend and/or frontend:")
        if not backend:
            print("  * --no-backend")
        if not frontend:
            print("  * --no-frontend")

    return backend, frontend


def start_frontend(backend, inference_service):
    model_manager = ModelManager.get_instance() if backend else None
    module_models = _get_module_models(model_manager)
    # Channel and stub is for client
    port = (
        get_config().runtime.port if not backend else backend.port
    )  # Using the actual port when we have a backend
    target = f"localhost:{port}"
    channel = grpc.insecure_channel(target)
    frontend = get_frontend(channel, inference_service, module_models)
    print(f"▶️  Starting the frontend gradio UI with using backend target={target}")
    frontend.launch(share=False, show_tips=False)
    print("⏹️  Stopped")


def main() -> int:
    # Command-line args allow running backend/frontend separately
    backend, frontend = _parse_args()

    # inference_service is needed for both Caikit backend server and frontend UI
    inference_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
    )

    if backend:
        print("▶️  Starting the backend Caikit inference server...")
        with RuntimeGRPCServer() as backend:
            if frontend:
                start_frontend(backend, inference_service)  # and wait for termination
            else:
                # Block on backend when there is no waiting on frontend (in same process)
                try:
                    backend.server.wait_for_termination()
                except KeyboardInterrupt as e:
                    # This is the expected CTRL-C when blocking on the server only (hiding the stack trace)
                    print(
                        f"\n⏹️  Stopping the backend Caikit inference server due to {repr(e)}"
                    )

    elif frontend:
        start_frontend(backend, inference_service)

    else:
        print(
            "⚠️  Inference service generation completed, but no backend or frontend servers started due to command-line options."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
