# DeepSeek-R1 OpenAI-Compatible LLM Serving on Modal

This repository demonstrates how to deploy a DeepSeek-R1 model in an OpenAI-compatible format using Modal and vLLM. The setup allows you to serve LLM inference with GPU acceleration and scale as needed.

## Overview

The code provides:

1. A Modal app that serves **DeepSeek-R1-Distill-Qwen-32B** model via vLLM
2. OpenAI-compatible API endpoints for chat and completions
3. GPU-accelerated inference using H100 GPUs
4. Authentication middleware
5. Model weights management
6. Modal Volume integration for persistent storage

## Prerequisites

- Modal account and API key
- NVIDIA H100 GPU access
- Python 3.12
- pip
- **DeepSeek-R1-Distill-Qwen-32B model weights downloaded to a Modal Volume**

## Installation

1. Clone the repository
2. **Download the DeepSeek-R1-Distill-Qwen-32B model weights:**
   ```bash
   modal run download_llama.py --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --model-revision d66bcfc2f3fd52799f95943264f32ba15ca0003d
   ```
3. Install dependencies:
   ```bash
   pip install modal vllm fastapi huggingface_hub
   ```

## Running the Interface

To run the DeepSeek-R1 OpenAI interface:

