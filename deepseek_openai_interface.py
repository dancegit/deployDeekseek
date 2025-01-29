# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/llm-serving/vllm_inference.py"]
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM

import modal
import json
import os

CHAT_TEMPLATE = """{
    "messages": [
        {
            "role": "system",
            "content": "Act as an expert software developer.
            Take requests for changes to the supplied code.
            If the request is ambiguous, ask questions.

            Always reply to the user in english.

            Once you understand the request you MUST:
            1. Determine if any code changes are needed.
            2. Explain any needed changes.
            3. If changes are needed, output a copy of each file that needs changes."
        }
    ]
}"""

# Create a stub file that will be copied into the image
stub_dir = os.path.dirname(__file__)
template_path = os.path.join(stub_dir, "chat_template.json")
with open(template_path, "w") as f:
    f.write(CHAT_TEMPLATE)

vllm_image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.6.3post1", "fastapi[standard]==0.115.4")
    .add_local_file(template_path, "/root/chat_template.json"))

MODELS_DIR = "/llamas"
MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
MODEL_REVISION = "61dc97b922b13995e7f83b7c8397701dbf9cfd4c"  #the commit hash

try:
    volume = modal.Volume.from_name("llamas", create_if_missing=False).hydrate()
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")

app = modal.App("deepseek-on-modal-openai")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
TOKEN = "zRjUP9GqzZpZyfQk8Hm06Tbk5v5U1ZJdPxtMMvf8sLvuzl3"  # auth token. for production use, replace with a modal.Secret
OPENAI_API_KEY = "zRjUP9GqzZpZyfQk8Hm06Tbk5v5U1ZJdPxtMMvf8sLvuzl3"
MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function(
    image=vllm_image,
    gpu=modal.gpu.H100(count=N_GPU),
    container_idle_timeout=15, #15 seconds timeout if no request we idle
    timeout=1 * HOURS,
    allow_concurrent_inputs=1,
    volumes={MODELS_DIR: volume},
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
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext

    volume.reload()  # ensure we have the latest version of the weights

    # Load chat template as JSON directly
    with open('/root/chat_template.json', 'r') as f:
        chat_template_str = f.read()
        chat_template = json.loads(chat_template_str)

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    # security: CORS middleware for external requests
    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # security: inject dependency on authed routes
    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN and api_key.credentials != OPENAI_API_KEY:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.95,
        max_model_len=8096,
        enforce_eager=True,  # False=capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=chat_template_str,  # Pass the template string instead of dict
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config
