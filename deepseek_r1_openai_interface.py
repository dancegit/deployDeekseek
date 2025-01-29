import modal
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import secrets
from pathlib import Path
import vllm.entrypoints.openai.api_server as api_server
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

app = modal.App("deepseek-r1-openai-interface")

# Model configuration
MODEL_NAME = "deepseek-ai/deepseek-llm-67b-chat"
MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"
MODELS_DIR = "/models"
CACHE_DIR = "/root/.cache"

# Volume setup
volume = modal.Volume.from_name("deepseek-models", create_if_missing=True)

# CUDA configuration
CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
OS_VERSION = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{OS_VERSION}"

# Container image configuration
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .pip_install(
        "vllm==0.6.3post1",
        "fastapi[standard]==0.115.4",
        "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .entrypoint([])
)

@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    timeout=30 * 60,  # 30 minutes
)
def download_model():
    from huggingface_hub import snapshot_download
    
    print(f"🦙 downloading model from {MODEL_NAME}")
    
    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=MODELS_DIR,
    )
    
    volume.commit()
    print("🦙 model downloaded and cached")

def generate_token():
    return secrets.token_urlsafe(32)

# Generate a token for deployment
TOKEN = generate_token()

@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    gpu=modal.gpu.H100(count=4),  # DeepSeek needs significant GPU memory
    container_idle_timeout=5 * 60,  # 5 minutes
    timeout=24 * 60 * 60,  # 24 hours
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def serve():
    volume.reload()  # ensure we have the latest version of the weights
    
    web_app = FastAPI(
        title="DeepSeek-R1 OpenAI-compatible API",
        description="OpenAI-compatible API for DeepSeek-R1 running on Modal",
        version="0.0.1",
        docs_url="/docs",
    )

    # Security: CORS middleware for external requests
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security: Bearer token authentication
    http_bearer = HTTPBearer(
        scheme_name="Bearer Token",
        description="Authentication token required for access.",
    )

    async def is_authenticated(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
        if credentials.credentials != TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"username": "authenticated_user"}

    # Add authentication to all routes
    router = FastAPI(dependencies=[Depends(is_authenticated)])
    router.include_router(api_server.router)
    web_app.include_router(router)

    # Configure vLLM engine
    engine_args = AsyncEngineArgs(
        model=str(Path(MODELS_DIR) / MODEL_NAME),
        tensor_parallel_size=4,  # Use all 4 GPUs
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enforce_eager=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    model_config = get_model_config(engine)

    # Set up vLLM OpenAI-compatible server
    api_server.engine = engine
    api_server.model_config = model_config

    return web_app

def get_model_config(engine):
    """Get model configuration from engine."""
    import asyncio

    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    return model_config

if __name__ == "__main__":
    # Download model if needed
    download_model.remote()
    print(f"\nTo use this API with OpenAI clients, configure:")
    print(f"  OPENAI_API_KEY={TOKEN}")
    print(f"  OPENAI_BASE_URL=https://{modal.config.get_current_workspace()}--deepseek-r1-openai-interface-serve.modal.run")
    print("\nAPI documentation available at:")
    print(f"  https://{modal.config.get_current_workspace()}--deepseek-r1-openai-interface-serve.modal.run/docs")
    print("\nDeploy with: modal deploy deepseek_r1_openai_interface.py")
