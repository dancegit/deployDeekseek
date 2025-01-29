# DeepSeek-R1 OpenAI-Compatible LLM Serving on Modal

This repository demonstrates how to deploy a DeepSeek-R1 model in an OpenAI-compatible format using Modal and vLLM. The setup allows you to serve LLM inference with GPU acceleration and scale as needed.

## Overview

The code provides:

1. A Modal app that serves DeepSeek-R1 model via vLLM
2. OpenAI-compatible API endpoints
3. GPU-accelerated inference using H100 GPUs
4. Authentication middleware
5. Model weights management

## Prerequisites

- Modal account and API key
- NVIDIA H100 GPU access
- Python 3.12
- pip

## Installation

1. Clone the repository
2. Install dependencies:
