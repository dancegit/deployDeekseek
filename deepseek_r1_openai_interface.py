import modal
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import secrets
from pathlib import Path

app = modal.App("deepseek-r1-openai-interface")

# Define CUDA configuration
CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
OS_VERSION = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{OS_VERSION}"

# Define the container image with CUDA support
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(
        "cmake --build llama.cpp/build --config Release -j --clean-first"
    )
    .pip_install("llama-cpp-python[server]", "fastapi", "uvicorn", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .entrypoint([])
)

# Model configuration
MODEL_REPO_ID = "Qwen/Qwen2-0.5B-Instruct-GGUF"
MODEL_PATTERN = "*q8_0.gguf"
CACHE_DIR = "/root/.cache/llama.cpp"
model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)

@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=30 * 60,  # 30 minutes
)
def download_model():
    from huggingface_hub import snapshot_download
    
    print(f"🦙 downloading model from {MODEL_REPO_ID} if not present")
    
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=CACHE_DIR,
        allow_patterns=[MODEL_PATTERN],
    )
    
    model_cache.commit()
    print("🦙 model loaded")

def generate_token():
    return secrets.token_urlsafe(32)

# Generate a token for deployment
TOKEN = generate_token()

@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    gpu=modal.gpu.H100(count=1),
    container_idle_timeout=5 * 60,  # 5 minutes
    timeout=24 * 60 * 60,  # 24 hours
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def serve():
    web_app = FastAPI(
        title="DeepSeek-R1 OpenAI-compatible server",
        description="Run DeepSeek-R1 with an OpenAI-compatible interface on Modal",
        version="0.0.1",
        docs_url="/docs",
    )

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

    @web_app.post("/v1/completions")
    async def create_completion(prompt: str, max_tokens: int = 100, depends=Depends(is_authenticated)):
        model_path = str(Path(CACHE_DIR) / MODEL_PATTERN)
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=9999,  # Use all GPU layers
            n_ctx=8192,  # Increased context window
            n_threads=12,
        )
        
        output = llm(prompt, max_tokens=max_tokens)
        return {
            "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "object": "text_completion",
            "created": int(os.time()),
            "model": MODEL_REPO_ID,
            "choices": [
                {
                    "text": output['choices'][0]['text'],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(llm.tokenize(prompt)),
                "completion_tokens": len(llm.tokenize(output['choices'][0]['text'])),
                "total_tokens": len(llm.tokenize(prompt)) + len(llm.tokenize(output['choices'][0]['text']))
            }
        }

    return web_app

if __name__ == "__main__":
    # Download model if needed
    download_model.remote()
    print(f"Generated authentication token: {TOKEN}")
    print("Deploy with: modal deploy deepseek_r1_openai_interface.py")
