# Using Your Own GPU for Inference

**Who is this guide for?** 
- People that want to run containerized inference on a GPU
- People that have some experience with using containers
- People that have experience with configuring and using remote resources

**What are the guide limitations?** 
- It assumes you have the remote already setup with appropriate dependencies, i.e. NVIDIA GPU drivers, the Container Toolkit, 
  and either Docker or Podman
- It assumes that Ubuntu 22.04 or Ubuntu 24.04 LTS is the reference OS
- Some steps may need adjustment 

**What else do I need to know?**
- You will need root/sudo access for most setup steps
- The first place to check for issues is `nvidia-smi` output
- Make sure your GPU meets the minimum requirements before starting

For detailed software installation instructions, see:
- [NVIDIA Driver Installation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- [NVIDIA Container Toolkit Setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Installation](https://docs.docker.com/engine/install/ubuntu/) or [Podman Installation](https://podman.io/docs/installation)

---

## High-Level Overview

There are many ways you can setup inference on a remote GPU. 

We will go over two: Ollama and NVIDIA NIM.

### Option A: Ollama (Intermediate)
- Easier to set up and manage
- Runs many models out of the box
- Good for experimentation and development
- See [Ollama Setup Guide](#ollama-setup) below
- GPU requirements: Depends on the model selected, but lighter weight than NIMs and can go down to 8 GB of vRAM

### Option B: NVIDIA NIM (Advanced)
- More steps to setup but has better performance and optimization
- Many options for configuring deployment and model optimization
- Better for production use
- See [NIM Setup Guide](#nim-setup) below
- GPU requirements: Depends on the model selected, but generally require 24GB of vRAM or higher

## General Prerequisites

- Make sure the remote is properly setup and that you have SSH access to it
  - IP address: ``<remote-ip>``
  - Remote user: ``<remote-user>``
- Be in a terminal session on the remote
- Make sure it's open to TCP access on a known port, i.e. ``<remote-port>``
- Make sure that the container runtime is properly configured

# Option A: Using Ollama

### Prerequisites for Ollama
- Satisfy the [General Prerequisites](#general-prerequisites)

### Three Basic Steps
- **Deploy Ollama Container**: Pull the Ollama container onto the remote and run it
- **Pull Model into Ollama Container**: Exec into the container and load the desired model
- **Add Ollama Container as an Endpoint**: Configure the Agentic RAG app to use the model

## Deploy Ollama Container

Do the following in the **remote** terminal. If you encounter any errors, use an LLM to help you debug. 

```bash
# Pull the Ollama container. Change the tag if you want a different one. 

docker pull ollama/ollama:latest

# Run it and make sure to connect the port on the remote, <remote-port>, to the Ollama port in the container, 11434

docker run -d --name ollama \
  --gpus all --restart unless-stopped \
  -p <remote-port>:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# Make sure it is running properly

curl http://localhost:10000/api/tags

```

## Pull Model into Ollama Container

Do the following in the **remote** terminal. If you encounter any errors, use an LLM to help you debug. 

```bash
# Exec into the container and pull a model

docker exec ollama ollama pull llama2:7b

# Verify the model has been pulled and is available

curl http://localhost:<remote-port>/api/tags

# Submit a simple query to the model to test

curl -X POST http://localhost:<remote-port>/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "prompt": "What is the meaning of life?",
    "stream": false
  }'


```

## Add Ollama Container as an Endpoint

Do the following in a **local** terminal. If you encounter any errors, use an LLM to help you debug. 

You will need the remote ip, ``<remote-ip>``, and the remote port, ``<remote-port>``.

```bash

# Test local access to the Ollama container on the remote 

curl -X POST http://<remote-ip>:<remote-port>/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:7b", "prompt": "Hello, Ollama!"}'
```



1. Open the **Agentic RAG** project in AI Workbench
2. Select **Models**
3. For each component you want to use the Ollama container:
   1. Select **Self-Hosted Endpoint**
   2. Enter the ip address, port and model name



# Option B: Using NVIDIA NIM

### Prerequisites for NVIDIA NIM

- Satisfy the [General Prerequisites](#general-prerequisites)
- Ensure you have an NVIDIA NGC API key (available at https://ngc.nvidia.com/setup/api-key)

### Three Basic Steps
- **Deploy NIM Container**: Pull and run the NIM container on the remote
- **Configure NIM Container**: Set up the model and API key
- **Add NIM Container as an Endpoint**: Configure the Agentic RAG app to use the model

## Deploy NIM Container

Do the following in the **remote** terminal. If you encounter any errors, use an LLM to help you debug. 

```bash
# Login to NVIDIA NGC
# Set user as $oauthtoken and password as the NGC key
docker login nvcr.io

# Pull the NIM container
docker pull nvcr.io/nim/deepseek-r1-distill-llama-8b:latest

# Run the container
docker run -d --name nim \
  --gpus all --restart unless-stopped \
  -p <remote-port>:8000 \
  -e NGC_API_KEY=<your-ngc-api-key> \
  nvcr.io/nim/deepseek-r1-distill-llama-8b:latest

# Verify the container is running
docker ps | grep nim
```

## Configure NIM Container

The NIM container will automatically download and configure the model when it starts. You can verify it's working by:

```bash
# Check the container logs

docker logs nim

# Test the API endpoint

curl -X POST http://localhost:<remote-port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, NIM!"}
    ]
  }'
```

## Add NIM Container as an Endpoint

Do the following in a **local** terminal. If you encounter any errors, use an LLM to help you debug. 

You will need the remote ip, ``<remote-ip>``, and the remote port, ``<remote-port>``.

```bash
# Test local access to the NIM container on the remote

curl -X POST http://<remote-ip>:<remote-port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, NIM!"}
    ]
  }'
```

1. Open the **Agentic RAG** project in AI Workbench
2. Select **Models**
3. For each component you want to use the NIM container:
   1. Select **Self-Hosted Endpoint**
   2. Enter the ip address, port and model name

### Additional Notes

- NIM containers are optimized for production use and provide better performance than Ollama
- They require more GPU memory (typically 24GB or more)
- The API follows the OpenAI-compatible format, making it easy to integrate with existing applications
- You can monitor the container's performance using NVIDIA's monitoring tools
- For production deployments, consider setting up proper security measures and load balancing
