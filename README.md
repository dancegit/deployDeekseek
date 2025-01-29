# DeepSeek-R1 OpenAI-Compatible LLM Serving on Modal

This repository demonstrates how to deploy a DeepSeek-R1 model in an OpenAI-compatible format using Modal and vLLM. The setup allows you to serve LLM inference with GPU acceleration and scale as needed.

## Overview

The code provides:

1. A Modal app that serves **DeepSeek-R1** model via vLLM
2. OpenAI-compatible API endpoints
3. GPU-accelerated inference using H100 GPUs
4. Authentication middleware
5. Model weights management

## Prerequisites

- Modal account and API key
- NVIDIA H100 GPU access
- Python 3.12
- pip
- **DeepSeek-R1 model weights downloaded to a Modal Volume**

## Installation

1. Clone the repository
2. **Download the DeepSeek-R1 model weights:**
   ```bash
   modal run download_llama.py --model-name deepseek-ai/DeepSeek-R1 --model-revision 5dde110d1a9ee857b90a6710b7138f9130ce6fa0
   ```
3. Install dependencies:

## DeepSeek-R1 OpenAI Interface

The `deepseek_r1_openai_interface.py` script provides an interface to serve the DeepSeek-R1 model in an OpenAI-compatible format. Here's what it does:

- **Model Serving**: It uses vLLM to serve the DeepSeek-R1 model, which is now the default model instead of LLaMA 3.1-8B.
- **Authentication**: Includes authentication middleware to secure the API endpoints.
- **GPU Utilization**: Configured to use NVIDIA H100 GPUs for inference, with settings for optimal GPU memory utilization.
- **Model Weights**: The script assumes the model weights are already downloaded to a Modal Volume named "llamas".

### Running the Interface

To run the DeepSeek-R1 OpenAI interface:

