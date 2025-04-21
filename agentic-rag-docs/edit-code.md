# Modifying the Agentic RAG Project

> **Who is this for?**  Anyone who wants to adapt or improve the application, e.g. by adding more endpoints, changing the endpoint provider, are modifying the Gradio app.


## 🧩 How to Add a New LLM by Adding an NVIDIA Endpoint to the UI

This section is on adding an NVIDIA-hosted model to the dropdown API menus in the **Models** tab of the Gradio interface (`converse.py`).

---
### Files and sections you will need to edit
- ``code/pages/converse.py``
    - Search: ``Model identifiers with prefix``
    - Search: ``Modify model identifiers``
    - Search: ``build_page()`` > ``model_list``


### 1. Find a model you want to add on build.nvidia.com

> We will use NVIDIA's [Llama 3.1 Nemotron Ultra](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1).

- Go to [build.nvidia.com](https://build.nvidia.com/models) and find a large language model, e.g. [Llama 3.1 Nemotron Ultra](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1)
- Copy the provider-model path, ``nvidia/llama-3_1-nemotron-ultra-253b-v1``


### 2. Add a Model Identifier

> Do this in ``code/pages/converse.py``. Search for ``Model identifers with prefix``.
> You should already see some models there:


```python
LLAMA = "meta/llama3-70b-instruct"
MISTRAL = "mistralai/mixtral-8x22b-instruct-v0.1"
```

Add your model here:

```python
NEMO = "nvidia/nemo-llama3-8b-instruct"
```

If you're using `INTERNAL_API` to route through a proxy or gateway:

```python
if INTERNAL_API != '':
    NEMO = f'{INTERNAL_API}/nvidia/nemo-llama3-8b-instruct'
```

---

## 2. Add to Dropdown Model List

Locate this line in `build_page()`:

```python
model_list = [LLAMA, MISTRAL]
```

Update it:

```python
model_list = [LLAMA, MISTRAL, NEMO]
```

This makes your new model available in all dropdowns (Router, Generator, etc.).

---
