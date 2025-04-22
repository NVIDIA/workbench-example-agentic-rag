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

### Option A: Ollama (Simpler)
- Easier to set up and manage
- Runs many models out of the box
- Good for experimentation and development
- See [Ollama Setup Guide](#ollama-setup) below
- GPU requirements: Depends on the model selected, but lighter weight than NIMs and can go down to 8 GB of vRAM

### Option B: NVIDIA NIM (More Advanced)
- More steps to setup but has better performance and optimization
- Many options for configuring deployment and model optimization
- Better for production use
- See [NIM Setup Guide](#nim-setup) below
- GPU requirements: Depends on the model selected, but generally require 24GB of vRAM or higher



## Option A: Using Ollama

#### Prerequisites

- Make sure the remote is properly setup and that you have SSH access to it
- Be in a terminal session on the remote
- Make sure it's open to TCP access on a known port, i.e. ``<remote_port>``
- Make sure that the container runtime is properly configured

#### Three Basic Steps
- **Deploy Ollama Container**: Pull the Ollama container onto the remote and run it
- **Pull Model into Ollama Container**: Exec into the container and load the desired model
- **Add Ollama Container as an Endpoint**: Configure the Agentic RAG app to use the model

### Deploy Ollama Container


```bash
# Pull the Ollama container
docker pull ollama/ollama:latest

# Run it
docker run -d --name ollama \
  --gpus all --restart unless-stopped \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest
```

### Pull Model into Ollama Container

```bash
# Pull a model through the container
docker exec ollama ollama pull llama2:7b

# Or pull Mistral 7B
docker exec ollama ollama pull mistral:7b
```

### Add Ollama Container as an Endpoint

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:7b", "prompt": "Hello, Ollama!"}'
```



1. Open the **Agentic RAG** project in AI Workbench
2. Go to **Config → Endpoints**
3. For each pipeline component:
   1. Click **Add Endpoint → Custom**
   2. Enter `http://localhost:11434`
   3. Select the model you pulled

--

## Option B: Using NVIDIA NIM

### TBD