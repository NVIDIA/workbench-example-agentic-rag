# Self‑Hosting Agentic RAG on Your Own GPU

> **Who is this for?**  Anyone who wants faster inference, air‑gapped privacy, or the ability to run larger models than the free cloud endpoints provide.

---

## 1  Why self‑host?

| Benefit | What it means |
|---------|---------------|
| **Performance** | Cut latency and remove rate limits by serving models on‑prem or in your own cloud VM. |
| **Data control** | Keep proprietary documents + queries inside your perimeter. |
| **Model choice** | Run any NIM‑packaged model (Llama 3, Mistral, etc.) or your own fine‑tune. |

---

## 2  Prerequisites

| Item | Minimum | Notes |
|------|---------|-------|
| NVIDIA GPU | RTX A6000 (48 GB) or >= 24 GB VRAM | Multiple smaller GPUs work; see *GPU sizing* below. |
| OS | Ubuntu 22.04 LTS | Other distros fine if drivers ≥ 550. |
| NVIDIA drivers | 550.xx or newer | `nvidia-smi` should show your GPU. |
| Docker Engine | 24.x | <https://docs.docker.com/engine/install/ubuntu> |
| NVIDIA Container Toolkit | `nvidia-container-toolkit` | Enables `--gpus all` flag. |
| Open ports | TCP 8000 (default) | Change with `-p` flag. |

> **Tip** If you are on bare‑metal Windows, create a Linux VM or dual‑boot. WSL 2 isn’t ideal for NIM GPU passthrough.

---

## 3  One‑time host setup

```bash
# 1 Install driver + toolkit (Ubuntu example)
sudo apt-get update && sudo apt-get install -y nvidia-driver-550
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2 Test
nvidia-smi       # should list your GPU
docker run --rm --gpus all nvidia/cuda:12.5.0-base nvidia-smi
```

---

## 4  Deploy a NIM container

Pick a model from the [NIM catalog](https://catalog.ngc.nvidia.com/orgs/nim) that fits your VRAM. Example: Llama 3 8B Instruct.

```bash
# Pull the container
sudo docker pull nvcr.io/nim/meta/llama3-8b-instruct:latest

# Run it (maps HTTP port 8000 on host → 8000 in container)
sudo docker run -d --name llama3_nim \
  --gpus all --restart unless-stopped \
  -p 8000:8000 nvcr.io/nim/meta/llama3-8b-instruct:latest
```

### 4.1  Verify the endpoint

```bash
curl -X POST http://<host-ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, NIM!","max_tokens":5}'
```
A JSON response means you’re good to go.

---

## 5  Wire Workbench to your NIM

1. Open the **Agentic RAG** project in AI Workbench.
2. Go to **Config → Endpoints**.
3. For each pipeline component you want to self‑host (Generator, Retriever, etc.):
   1. Click **Add Endpoint → NIM**.
   2. Enter `http://<host-ip>:8000` (adjust port if changed).
   3. Select the model name that matches your container tag.
4. Click **Save** and restart **Chat**.

Workbench will validate GPU compatibility automatically. Components you leave unchanged will continue to use the default cloud endpoints, so you can mix & match.

---

## 6  GPU sizing cheat‑sheet

| Model | Approx VRAM | Good for |
|-------|-------------|----------|
| Llama 3 8B Instruct | ≈ 16 GB | Chat UI, experimentation |
| Mistral 7B | ≈ 8 GB | Small instances, edge devices |
| Llama 3 70B | ≈ 48 GB (2× 24) | Production Q&A, long answers |

> **Rule of thumb** VRAM ≈ ( model size × 1.3 ) + headroom for batch.

---

## 7  Troubleshooting

| Symptom | Fix |
|---------|-----|
| **`docker: Error response from daemon: could not select device driver`** | Reinstall `nvidia-container-toolkit`; ensure `--gpus all` support. |
| **Endpoint 404 / no route** | Check container logs: `docker logs llama3_nim`. Ensure `-p` flag matches Workbench URL. |
| **OOM / CUDA out‑of‑memory** | Choose smaller model or add `--gpus device=0,1` to span multiple GPUs. |

---

## 8  Security best practices

- Bind to localhost and reverse ‑ proxy via Nginx with SSL if exposed to the internet.
- Set `--restart unless-stopped` so the service auto‑starts on reboot.
- Use firewall rules (`ufw allow from <WB-IP> to any port 8000`).

---

## 9  Cleaning up

```bash
sudo docker stop llama3_nim && sudo docker rm llama3_nim
sudo docker image rm nvcr.io/nim/meta/llama3-8b-instruct:latest
```

That’s it—Workbench will now drive your own GPU‑powered model just like the default cloud endpoints.

