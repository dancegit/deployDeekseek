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
            "content": "Act as an expert software developer that should: \\n
                \\n
                - Help with code writing, debugging, and understanding. \\n
                - Use markdown code blocks for all code snippets (` ```python`, etc).  \\n
                - Provide explanations or comments for code when necessary. \\n
                - Respond with professionalism but keep the tone helpful and slightly humorous. \\n
                - Acknowledge when files are added or removed from the session. \\n
                - Use colorized "edit blocks" to suggest code modifications if possible. \\n
                \\n
                You MUST:\\n
                1. Determine if any code changes or text updates are needed.\\n
                2. Explain any needed changes.\\n
                3. If changes are needed, output a copy of each file that needs changes.\\n
                \\n
                To suggest changes to a file you MUST return the entire content of the updated file.\\n
                You MUST use this *file listing* format according to this example (but could be any file):\\n
                \\n
                path/to/filename.js\\n
                ```\\n
                // entire file content ...\\n
                // ... goes in between\\n
                ```\\n
                \\n
                Every *file listing* MUST use this format:\\n
                - First line: the filename with any originally provided path; no extra markup, punctuation, comments, etc. **JUST** the filename with path.\\n
                - Second line: opening ```\\n
                - ... entire content of the file ...\\n
                - Final line: closing ```\\n
                \\n
                To suggest changes to a file you MUST return a *file listing* that contains the entire content of the file.\\n
                *NEVER* skip, omit or elide content from a *file listing* using \\\"...\\\" or by adding comments like \\\"... rest of code...\\\"!\\n
                Create a new file you MUST return a *file listing* which includes an appropriate filename, including any appropriate path."
        }
    ]
}"""

vllm_image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("nm-vllm==0.6.3post1", "fastapi[standard]==0.115.4"))

MODELS_DIR = "/llamas"
MODEL_NAME="neuralmagic-ent/Llama-3.3-70B-Instruct-quantized.w8a8"
MODEL_REVISION = "dc36722e6cb1e6b98d0144fd6059933d19c00ebf"


#MODEL_NAME="mradermacher/oh-dcft-v3.1-claude-3-5-haiku-20241022-qwen-GGUF"
#MODEL_REVISION="7a01304dd9537438e6b161b0f03cc4e2d14bd324"


#MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
#MODEL_REVISION = "2453c79a2a0947968a054947b53daa598cb3be52"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#MODEL_REVISION = "1772b078b94935926dcc8715c1afdd04ae447080"


#MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
#MODEL_REVISION = "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"

#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#MODEL_REVISION = "d66bcfc2f3fd52799f95943264f32ba15ca0003d" #the commit hash

#MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
#MODEL_REVISION = "61dc97b922b13995e7f83b7c8397701dbf9cfd4c"  #the commit hash




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
    gpu=modal.gpu.A100(count=N_GPU,size="80GB"),  #L40S HAS 48GB , L4 HAS 24GB T4 HAS 16GB,
    container_idle_timeout=2*MINUTES, #5 seconds timeout if no request we idle to save money
    timeout=1 * HOURS,
    allow_concurrent_inputs=1,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve():
    import fastapi
    import nm_vllm.entrypoints.openai.api_server as api_server
    from nm_vllm.engine.arg_utils import AsyncEngineArgs
    from nm_vllm.engine.async_llm_engine import AsyncLLMEngine
    from nm_vllm.entrypoints.logger import RequestLogger
    from nm_vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from nm_vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from nm_vllm.entrypoints.openai.serving_engine import BaseModelPath
    from nm_vllm.usage.usage_lib import UsageContext

    volume.reload()  # ensure we have the latest version of the weights

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
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # False=capture the graph for faster inference, but slower cold starts (30s > 20s)
        trust_remote_code=True,  # Add this line to trust remote code
        quantization=None,  # nm-vllm will handle this; set to None to avoid conflicts
    )
    # Remove the following line as nm-vllm should manage quantization:
    # os.environ['VLLM_QUANTIZATION'] = 'w8a8'
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)
    # Here, you don't need to set quantization_config manually

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=None,  # Use CHAT_TEMPLATE directly
        response_role="system",
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
