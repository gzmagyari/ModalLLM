# Modal LLM Serving Project

## Project Overview

This project demonstrates how to run a Large Language Model (LLM) on [Modal](https://modal.com), a cloud platform for running serverless applications. The project includes three main components:

1. Downloading model weights from Hugging Face
2. Serving an LLM using a Modal volume
3. Serving an LLM with model weights baked into the Docker image

The project uses [vLLM](https://docs.vllm.ai/en/latest/), an open-source library for LLM inference, to create an OpenAI-compatible API server.

## File Descriptions

### 1. downloadModel.py

This script downloads a specified model from Hugging Face and stores it in a Modal volume. It uses the `huggingface_hub` library for efficient downloads.

Key features:

- Downloads model weights to a Modal volume
- Uses `hf-transfer` for faster downloads
- Configurable model name (default: "bluuwhale/L3-SthenoMaidBlackroot-8B-V1")

### 2. modalAppWithVolume.py

This file sets up a vLLM server in OpenAI-compatible mode, serving the model from a Modal volume.

Key features:

- Uses FastAPI to create an OpenAI-compatible API
- Implements simple authentication middleware
- Configurable GPU count and type
- Serves both chat and completion endpoints

### 3. modalApp.py

Similar to `modalAppWithVolume.py`, but instead of using a Modal volume, it bakes the model weights directly into the Docker image.

Key features:

- Downloads model weights during image build time
- Eliminates the need for a separate volume
- Potentially faster cold start times
- Otherwise similar functionality to `modalAppWithVolume.py`

## Installation

To use this project, you need to have a Modal account and the Modal CLI installed. Follow these steps:

1. Install the Modal CLI:

   ```
   pip install modal
   ```

2. Authenticate with Modal:

   ```
   modal token new
   ```

3. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

## Usage

### Downloading the Model

To download the model weights to a Modal volume:

```
modal run downloadModel.py
```

This step is necessary before running `modalAppWithVolume.py`.

### Running the LLM Server with Volume

To deploy the LLM server using the Modal volume:

```
modal deploy modalAppWithVolume.py
```

### Running the LLM Server with Baked-in Weights

To deploy the LLM server with model weights baked into the Docker image:

```
modal deploy modalApp.py
```

After deployment, you'll receive a URL for your API endpoint. You can access the Swagger UI documentation at the `/docs` route of this URL.

## Additional Information

- The project uses Meta's LLaMA 3.1 8B Instruct variant by default.
- Authentication is implemented using a simple token-based system. For production use, replace the token with a Modal Secret.
- The server supports both chat and completion endpoints, compatible with the OpenAI API format.
- You can adjust the GPU count and type in both `modalAppWithVolume.py` and `modalApp.py` to optimize for your specific use case.
- For interacting with the API programmatically, you can use the Python `openai` library. Refer to the `client.py` script in the Modal examples repository for usage examples.
- A basic load-testing setup using `locust` is available in the `load_test.py` script in the Modal examples repository.

For more detailed information on Modal and vLLM, refer to their respective documentation:

- [Modal Documentation](https://modal.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
