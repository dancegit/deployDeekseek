import modal
import os
from pathlib import Path
import secrets
import subprocess
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

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
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands(
        "CMAKE_ARGS='-DGGML_CUDA=ON' pip install 'llama-cpp-python[server]'"
    )
    .pip_install("huggingface_hub[hf_transfer]")
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
    
    # Start llama.cpp server with our configuration
    model_path = str(Path(MODELS_DIR) / MODEL_NAME)
    server_args = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--n_gpu_layers", "-1",  # Use all GPU layers
        "--n_ctx", "8192",
        "--chat_format", "chatml",
        "--n_threads", "12"
    ]
    
    process = subprocess.Popen(server_args)
    
    # Create FastAPI app for authentication wrapper
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

    # Add authentication middleware
    @web_app.middleware("http")
    async def auth_middleware(request, call_next):
        if request.url.path in ["/docs", "/openapi.json"]:
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = auth_header.split(" ")[1]
        if token != TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return await call_next(request)

    return web_app

if __name__ == "__main__":
    # Download model if needed
    download_model.remote()
    print(f"\nTo use this API with OpenAI clients, configure:")
    print(f"  OPENAI_API_KEY={TOKEN}")
    print(f"  OPENAI_BASE_URL=https://{modal.config.get_current_workspace()}--deepseek-r1-openai-interface-serve.modal.run")
    print("\nAPI documentation available at:")
    print(f"  https://{modal.config.get_current_workspace()}--deepseek-r1-openai-interface-serve.modal.run/docs")
    print("\nDeploy with: modal deploy deepseek_r1_openai_interface.py")
