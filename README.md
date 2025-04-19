<img src="./readme-images/static/agentic-rag-screen-shot.png" width="80%" height="auto">

*Navigating the README:* [Get Started](#get-started) | [Easy Mode Details](#easy-mode-details) | [Advanced Mode Details](#advanced-mode-details) | [Repository License](#license)

<!-- Links -->
*Other Resources:* [:arrow_down: Download AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) | [:book: User Guide](https://docs.nvidia.com/ai-workbench/) |[:open_file_folder: Other Projects](https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html) | [:rotating_light: User Forum](https://forums.developer.nvidia.com/t/support-workbench-example-project-agentic-rag/303414)

# Agentic RAG - Control RAG for accuracy and hallucination

### Boost RAG with an agentic layer that:
- routes queries and adds live web search context is thin
- grades responses for relevance and accuracy, flags hallucinations
- lets you edit prompts for every stage from the UI

### Run inference your way:
- **Free out-of-the-box**: use free endpoints on build.nvidia.com
- **Self-hosted**: Point to NIM or other models on your own GPUs


## Get Started 

### Prerequisites
This app runs in [NVIDIA AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/), a free, cloud-like UX that runs on your own systems.
To follow this readme, you must first install [AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/overview.html) on your laptop.

### Easy Mode (5 minutes or less if Workbench already installed)
1. Get NVIDIA and Tavily API keys:  
   - ``NVIDIA_API_KEY`` → [Generate](https://org.ngc.nvidia.com/setup/api-keys)  See instructions [here](https://docs.nvidia.com/ai-enterprise/deployment/spark-rapids-accelerator/latest/appendix-ngc.html).
   - ``TAVILY_API_KEY`` → [Generate](https://tavily.com)  
2. **Clone** this repo with AI Workbench > [configure the keys](https://docs.nvidia.com/ai-workbench/user-guide/latest/environment/variables.html#basic-usage-for-environment-variables) when prompted.  
3. Click **Open Chat** > Go to the **Document** tab in the web app > Click **Add to Context**.  
4. Type in your question > Hit enter  - the answers come from free cloud endpoints.

### Advanced Mode (need to self-host GPUs)

1. Set up a Linux box with an NVIDIA GPU and Docker.  
2. Deploy an **NVIDIA NIM** container on that host.  
3. Configure the chat app to use the NIM.

See [full instructions here](agentic-rag-docs/self-host.md).


## Easy Mode Details 
Follow these steps if you want to get up and running but don't care about modifying the application or adding your own endpoints.

### First: Clone this project > Start the chat >  Create the context >  Ask your questions

| Step | Screenshot | What can go wrong |
|------|------------|-------------------|
| 1. Open the Desktop App > Select [local](https://docs.nvidia.com/ai-workbench/user-guide/latest/locations/locations.html). | <img src="./readme-images/desktop-icon.png" width="120" height="auto"> | Probably a  Docker Desktop issue (if selected on install). **Fix**:  Make sure you're logged in to Docker Desktop. See [troubleshooting here](https://docs.nvidia.com/ai-workbench/user-guide/latest/troubleshooting/troubleshooting.html) | 
| 2. Click **Clone Project** > Paste repository [URL](https://github.com/NVIDIA/workbench-example-agentic-rag) > **Clone** | <img src="./readme-images/clone-button.png" width="250" height="auto"> | Incorrect URL. **Fix**: use the correct URL. | 
| 3. Click **Resolve Now** > Enter NVIDIA and Tavily API keys. | <img src="./readme-images/resolve-now.png" width="200" height="auto"> | You miss the banner. **Fix**: go to **Project Container > Variables > Configure** for API keys. See [docs here](https://docs.nvidia.com/ai-workbench/user-guide/latest/environment/variables.html) | 
| 4. Click **Open Chat**. | <img src="./readme-images/open-chat-screen-shot.png" width="250" height="auto"> | Very little can go wrong here |
| 5. Click **Documents > Create Context**. | <img src="./readme-images/add-to-context-button.png" width="300" height="auto"> | Incorrect API key. Fix per Step 3 above. | 
| 6. Ask your question > Hit  enter. | <img src="./readme-images/hit-enter.png" width="200" height="auto"> | Incorrect API key. Fix per Step 3 above. | 

### Then: Clear the context > Change the URLs > Create the context > Ask your questions

| Step | What happens | What can go wrong | 
|------|--------------|-------------------|
| 1. Click **Documents > Clear Context**. | Resets vector database. | Very little.
| 2. Delete the URLs > Add your own > Click **Add to Context**. | Creates a new context. |  URLs that can't be resolved. **Fix**: Enter appropriate URLs | 
| 3. Ask your question > Hit enter. | Triggers the agent. | Incorrect API key. **Fix**: Fix per Step 3 in table above. | 


## Advanced Mode Details

<img src="./code/chatui/static/agentic-flow.png" width="100%" height="auto">

Use these details if you want to modify the application, e.g. by adding your own endpoints, changing the Gradio app or whatever else occurs to you.

### First: Fork this repo to your GitHub account > Clone it in Workbench 
This repository is read-only, so if you want to customize this app and share the changes, then you should [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) this repository before you clone it. 





 




#### Using Self-Hosted Endpoints

You can configure pipeline components (Router, Generator, Retrieval, Hallucination Check, Answer Check) separately to use either an endpoint or a self-hosted NIM, as well as selecting different models. In otherwords, you can mix and match between hosted and self-hosted components based on your needs. The application includes built-in GPU compatibility checking **for the remote host GPUs** to help you select appropriate models for that hardware configuration.

Prerequisites for the remote GPU:
* NVIDIA GPU(s) with appropriate VRAM
* Ubuntu 22.04 or later with latest NVIDIA drivers
* Docker and NVIDIA Container Toolkit

To set up NIM endpoints for your components:
1. Check the [NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) for detailed setup instructions
2. For each component you want to self-host:
   * Select "NIM Endpoints" in the component's configuration
   * Choose your GPU type and count - the UI will automatically show only compatible models
   * Enter your endpoint details (host, port)
3. Components not set to self-hosted will continue using their configured cloud endpoints

The application will validate your GPU configuration for each component and prevent incompatible model selections. You can use different GPU configurations for different components based on their computational needs.


# License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/NVIDIA/workbench-example-agentic-rag/blob/main/LICENSE.txt)

This project may utilize additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 

| :question: Have Questions?  |
| :---------------------------|
| Please direct any issues, fixes, suggestions, and discussion on this project to the DevZone Members Only Forum thread [here](https://forums.developer.nvidia.com/t/support-workbench-example-project-agentic-rag/303414) |
