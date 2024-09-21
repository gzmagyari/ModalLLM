# ---
# deploy: true
# cmd: ["modal", "serve", "vllm_inference.py"]
# pytest: false
# ---
# # Run an OpenAI-Compatible vLLM Server with Baked-in Model Weights
#
# This example demonstrates how to run a vLLM server in OpenAI-compatible mode on Modal, with the model weights baked into the container image itself. This approach eliminates the need for a separate volume to store the model weights and can simplify deployment and cold start times.
#
# We will use Meta's LLaMA 3.1 8B in the Instruct variant that's trained to chat and follow instructions.

import os
import modal

# Define the model directory and name
MODEL_DIR = "/model"
MODEL_NAME = "bluuwhale/L3-SthenoMaidBlackroot-8B-V1"

# Function to download the model weights into the image during build time
def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        use_auth_token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

# Define the container image with the baked-in model
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.5.3post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App("vllm-openai-compatible")

# Configuration variables
N_GPU = 1  # Adjust based on your model size and GPU availability
TOKEN = "super-secret-token"  # Authentication token; replace with a modal.Secret in production
MINUTES = 60  # Seconds
HOURS = 60 * MINUTES

# Define the ASGI app function
@app.function(
    image=vllm_image,
    gpu=modal.gpu.A100(count=N_GPU, size="40GB"),
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.usage.usage_lib import UsageContext

    # Create a FastAPI app using vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on Modal",
        version="0.0.1",
        docs_url="/docs",
    )

    # Security: CORS middleware for external requests
    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="Provide a valid authentication token.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security: Inject dependency on authenticated routes
    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    # Include vLLM's router within the authenticated router
    router.include_router(api_server.router)
    web_app.include_router(router)

    # Define engine arguments
    engine_args = AsyncEngineArgs(
        model=os.path.join(MODEL_DIR, MODEL_NAME),
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # Set to True for faster inference but slower cold starts
    )

    # Initialize the vLLM engine
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    # Retrieve model configuration
    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    # Set up OpenAI-compatible chat and completion services
    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        chat_template=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app

# Utility function to get model configuration
def get_model_config(engine):
    import asyncio

    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If there's an existing event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # If no event loop is running
        model_config = asyncio.run(engine.get_model_config())

    return model_config
