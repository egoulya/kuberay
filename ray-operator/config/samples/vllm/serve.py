import os
import logging
from typing import Dict, Optional, List

# Apply monkey-patches BEFORE importing vLLM!
import ray
from ray import serve

# Monkey-patch: Make Ray think GPUs exist even if not labeled "GPU"
_original_available_resources = ray.available_resources

import ray._private.state as ray_state

_original_nodes = ray_state.nodes

def patched_ray_state_nodes(*args, **kwargs):
    nodes = _original_nodes(*args, **kwargs)
    for node in nodes:
        if node.get("Alive", False):
            resources = node["Resources"]
            if "GPU" not in resources:
                grouped_gpu_keys = [k for k in resources if k.startswith("GPU_group")]
                total_gpus = sum(resources[k] for k in grouped_gpu_keys)
                if total_gpus > 0:
                    resources["GPU"] = total_gpus
                    print(f">>> [FAKE GPU PATCH] Injected 'GPU': {total_gpus} in ray._private.state.nodes() for node {node['NodeManagerAddress']}")
    return nodes

ray_state.nodes = patched_ray_state_nodes

def patched_available_resources():
    resources = _original_available_resources()
    if "GPU" not in resources:
        grouped = [k for k in resources if k.startswith("GPU_group")]
        fake_gpu = sum(resources[k] for k in grouped)
        if fake_gpu > 0:
            resources["GPU"] = fake_gpu
            print(f">>> [FAKE GPU PATCH] Injected GPU={fake_gpu} in available_resources()")
    return resources

ray.available_resources = patched_available_resources

# Monkey-patch vLLM's GPU check in initialize_ray_cluster
from vllm.executor import ray_utils
_original_init_cluster = ray_utils.initialize_ray_cluster

def patched_initialize_ray_cluster(parallel_config):
    print(">>> [PATCH ACTIVE] Skipping vLLM initialize_ray_cluster GPU check")
    from vllm.executor.parallel_utils import get_parallel_config
    from vllm.executor.executor import RayExecutor

    ray.init(address="auto", ignore_reinit_error=True)
    parallel_config = get_parallel_config(parallel_config)
    RayExecutor.initialize_parallel_groups(parallel_config)

ray_utils.initialize_ray_cluster = patched_initialize_ray_cluster

import ray._private.state as ray_state
_original_node_resources = ray_state.node_resources

def patched_node_resources(node_id: Optional[str] = None):
    resources = _original_node_resources(node_id)
    if "GPU" not in resources:
        grouped = [k for k in resources if k.startswith("GPU_group")]
        fake_gpu = sum(resources[k] for k in grouped)
        if fake_gpu > 0:
            resources["GPU"] = fake_gpu
            print(f">>> [FAKE GPU PATCH] Injected 'GPU': {fake_gpu} in ray_state.node_resources() for node {node_id}")
    return resources

ray_state.node_resources = patched_node_resources

# Now safe to import rest of vLLM
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        self.openai_serving_chat = None
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template

        # Post-init model init — defer engine until inside __init__
        self.engine_args = engine_args
        logger.info(f">>> Initializing AsyncLLMEngine with args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            model_names = self.engine_args.served_model_name or [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
            )

        logger.info(f">>> Chat request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(generator, media_type="text/event-stream")
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())

def parse_vllm_args(cli_args: Dict[str, str]):
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible API server")
    arg_parser = make_arg_parser(parser)
    args_list = [f"--{k}" if not k.startswith("--") else k for k in cli_args for _ in (0, 1)]
    for i, k in enumerate(cli_args):
        args_list[2 * i + 1] = str(cli_args[k])
    parsed_args = arg_parser.parse_args(args=args_list)
    return parsed_args

def build_app(cli_args: Dict[str, str]) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )

model = build_app({
    "model": os.environ["MODEL_ID"],
    "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
    "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
})
