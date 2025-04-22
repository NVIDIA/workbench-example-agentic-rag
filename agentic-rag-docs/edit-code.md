# Guide to Modifying the Agentic RAG Project

**Who is this guide for?** 
- People that know some Python
- People that want to adapt or improve the application
- People that want to explore code to see how agents work


**What are the guide limitations?** 
- It isn't comprehensive and doesn't go into full detail
- It assumes you can basically figure things out if you are pointed to the correct section

**What else do I need to know?**
- You will need to be able to check errors and do some debugging if you alter the code. Best to use an LLM to help you 
  interpret errors and figure out what to do.
- The first place to find errors is in the **Output** widget in the Desktop App (bottom left corner)
    - Click **Output** and select **Chat** from the dropdown. 



---

## 🧩 How to Add a New NVIDIA Endpoint to the Model Dropdown in the UI

You can add more NVIDIA endpoints to the dropdown API menus in the **Models** tab of the Gradio interface (`converse.py`).

You can add your model by:

- providing a new variable, e.g. ``NEMO``, assigned to the relevant endpoint string
- (NVIDIA only) adding that endpoint to the conditional logic that prepends the internal endpoint bits
- Updating the list of models pulled into the dropdown   

Adding a model this way will make it available to all of the different pipeline components.

### Files and sections you will need to edit
- ``code/chatui/pages/converse.py``
    - Search: ``Model identifiers with prefix``
    - Search: ``Modify model identifiers``
    - Search: ``build_page()`` > ``model_list``


### 1. Find a model you want to add on build.nvidia.com

- Go to [build.nvidia.com](https://build.nvidia.com/models) and find a large language model, e.g. [Llama 3.1 Nemotron Ultra](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1)
- Copy the provider-model path, e.g. ``nvidia/llama-3_1-nemotron-ultra-253b-v1``

### 2. Add a Model Identifier

- Find the ``Model identifers with prefix`` in ``code/chatui/pages/converse.py``
- Add your model to the section by defining a new variable:
    ```python
    NEMO = "nvidia/llama-3_1-nemotron-ultra-253b-v1"
    ```

### 3. Add Model to the Dropdown Model List

- Find the list ``model_list`` in the ``build_page()`` function
- Add your model to the list:
    ```python
    model_list = [LLAMA, MISTRAL, NEMO]
    ```

### 4. (NVIDIA only) Add Model to Internal Endpoint Logic

If you're using `INTERNAL_API`, you need to make sure you put the proper prefix on the model identifier.

Find the ``# Modify model identifiers`` section and update the endpoint logic:

```python
if INTERNAL_API != '':
    NEMO = f'{INTERNAL_API}/nvidia/llama-3_1-nemotron-ultra-253b-v1'
```

### Caveats
- The guidance below is to give you an idea of how to change things by adding a single model. You may want to add many models, 
  and if so the current code and the guidance below should be modified.
- For example, there are different Llama, Mistral, and NVIDIA model endpoints on build.nvidia.com. If you want to add more than one model from a given provider, 
  the naming convention used below would need to change.
- In addition, the internal endpoint logic is relevant to NVIDIA's internal endpoints, not necessarily to any other setup. 
  If you aren't at NVIDIA, it's not currently setup to support you.

## 🧩 How to Modify the Embedding Model

You can modify the embedding model used for document processing in the vector database by editing the configuration in `code/chatui/utils/database.py`.

### Files and sections you will need to edit
- ``code/chatui/utils/database.py``
    - Search: ``Default model for public embedding``
    - Search: ``Set the chunk size and overlap``

### 1. Choose Your Embedding Model

- Select an embedding model that is compatible with your needs
- Common choices include:
  - OpenAI's text-embedding-ada-002
  - Hugging Face's sentence-transformers
  - Cohere's embedding models
  - Or any other embedding model that provides vector representations

### 2. Modify the Embedding Model Configuration

- Find the ``Default model for public embedding`` section in ``code/chatui/utils/database.py``
- Update the ``EMBEDDINGS_MODEL`` variable with your chosen model:
    ```python
    # Default model for public embedding
    EMBEDDINGS_MODEL = 'your-embedding-model-name'
    ```

### 3. Optional: Adjust Chunk Size and Overlap

You can modify how documents are split and processed by adjusting the chunk size and overlap parameters:

```python
# Set the chunk size and overlap for the text splitter
DEFAULT_CHUNK_SIZE = 250  # Adjust this value to change chunk size
DEFAULT_CHUNK_OVERLAP = 0  # Adjust this value to change overlap
```

### Caveats
- The embedding model must be compatible with the vector store implementation
- Changing the embedding model will require re-embedding all documents in your vector store
- The chunk size and overlap settings affect how documents are processed and retrieved
- Make sure to update any API keys or authentication required for the new embedding model

## 🧩 How to Modify Vector Database Clearing Behavior

You can modify how the vector database is cleared by adjusting the `delete_all` parameter in the `_clear()` function in `code/chatui/utils/database.py`.

### Files and sections you will need to edit
- ``code/chatui/utils/database.py``
    - Search: ``Clear the Chroma collection``
    - Search: ``delete_all: bool = True``

### 1. Understand the Clearing Options

The vector database clearing has two behaviors:
- Basic clearing: Only clears the current Chroma collection
- Full clearing (default): Clears both the collection and all associated files/directories

### 2. Modify the Clear Function Parameter

- Find the ``_clear()`` function in ``code/chatui/utils/database.py``
- Change the default value of `delete_all` to `False` to preserve previous searches:
    ```python
    def _clear(
        persist_directory: str = "/project/data",
        collection_name: str = "rag-chroma",
        delete_all: bool = False  # Changed from True to False
    ):
    ```

### Caveats
- Setting `delete_all` to `False` will preserve files in the persist directory
- Hidden files (starting with '.') are always preserved regardless of this setting
- The current collection will still be cleared even with `delete_all = False`

## 🧩 How to Modify the Agent's Recursion Limit

You can modify how many times the agent can recursively process a query by adjusting the recursion limit in `code/chatui/pages/converse.py`.

### Files and sections you will need to edit
- ``code/chatui/pages/converse.py``
    - Search: ``Set recursion limit``
    - Search: ``DEFAULT_RECURSION_LIMIT``

### 1. Understand the Recursion Limit

The recursion limit controls how many times the agent can:
- Re-route questions between different components
- Re-try generating answers when previous attempts fail
- Iterate through the document retrieval and grading process

### 2. Modify the Recursion Limit

You can change the recursion limit in two ways. 

Regardless of how you set it, the app must be restarted for the change to take effect. 

1. **Environment Variable** (Recommended):
   - If the app is **not running**:
     ```bash
     # In a terminal attached to your container
     export RECURSION_LIMIT=20  # Set to your desired value
     # Then start the app
     ```
   - If the app is **already running**:
     ```bash
     # In a terminal attached to your container
     # First stop the app
     # Then set the environment variable
     export RECURSION_LIMIT=20
     # Then restart the app
     ```

2. **Direct Code Modification**:
   - Find the recursion limit configuration in `converse.py`:
   ```python
   DEFAULT_RECURSION_LIMIT = 10
   RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT))
   ```
   - Change the `DEFAULT_RECURSION_LIMIT` value to your desired number
   - This change will take effect after restarting the app

### Caveats
- Setting the limit too high may cause the agent to get stuck in loops
- Setting the limit too low may prevent the agent from fully processing complex queries
- The default value of 10 is a good balance for most use cases
- You can monitor recursion depth in the Output widget of the Desktop App
- Environment variable changes require app restart to take effect
- Code modifications require app restart and code recompilation to take effect
- Environment variables must be set in a terminal attached to the running container

## 🧩 How to Modify Tavily Search Settings

You can modify how many search results Tavily returns by adjusting the `TAVILY_K` parameter in `code/chatui/utils/graph.py`.

### Files and sections you will need to edit
- ``code/chatui/utils/graph.py``
    - Search: ``Tavily related parameters``
    - Search: ``DEFAULT_TAVILY_K``

### 1. Understand Tavily Search Results

The `TAVILY_K` parameter controls:
- How many search results are returned from Tavily
- The amount of web content available for the agent to process
- The breadth of information considered when answering questions

### 2. Modify the Search Results Limit

You can change the number of search results in two ways.

Regardless of how you set it, the app will need to be restarted.

1. **Environment Variable** (Recommended):
   - If the app is **not running**:
     ```bash
     # In a terminal attached to your container
     export TAVILY_K=5  # Set to your desired value
     # Then start the app
     ```
   - If the app is **already running**:
     ```bash
     # In a terminal attached to your container
     # First stop the app
     # Then set the environment variable
     export TAVILY_K=5
     # Then restart the app
     ```

2. **Direct Code Modification**:
   - Find the Tavily configuration in `graph.py`:
   ```python
   DEFAULT_TAVILY_K = 3
   TAVILY_K = int(os.getenv("TAVILY_K", DEFAULT_TAVILY_K))
   ```
   - Change the `DEFAULT_TAVILY_K` value to your desired number
   - This change will take effect after restarting the app

### Caveats
- Setting the value too high may increase response time and API costs
- Setting the value too low may limit the information available to the agent
- The default value of 3 is a good balance for most use cases
- Environment variable changes require app restart to take effect
- Code modifications require app restart and code recompilation to take effect
- Environment variables must be set in a terminal attached to the running container
