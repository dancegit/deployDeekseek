import modal
import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from llama_cpp import Llama
import secrets

app = modal.App("deepseek-r1-openai-interface")

# Define the container image with llama-cpp-python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llama-cpp-python", "fastapi", "uvicorn", "huggingface_hub")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands("cd llama.cpp && make")
)

# Model configuration
MODEL_REPO_ID = "Qwen/Qwen2-0.5B-Instruct-GGUF"
MODEL_FILENAME = "*q8_0.gguf"
MODEL_DIR = "/models"

@app.function(
    image=image,
    volumes={MODEL_DIR: modal.Volume.from_name("deepseek-r1", create_if_missing=True)},
    timeout=60 * 60,  # 1 hour
)
def download_model():
    from huggingface_hub import snapshot_download
    import os

    if not os.path.exists(f"{MODEL_DIR}/{MODEL_FILENAME}"):
        print(f"Downloading model from {MODEL_REPO_ID}")
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=MODEL_DIR,
            allow_patterns=[MODEL_FILENAME],
        )
    else:
        print("Model already exists in the volume.")

def generate_token():
    return secrets.token_urlsafe(32)

# Generate a token for deployment
TOKEN = generate_token()

@app.function(
    image=image,
    volumes={MODEL_DIR: modal.Volume.from_name("deepseek-r1")},
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
        llm = Llama(model_path=f"{MODEL_DIR}/{MODEL_FILENAME}")
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
