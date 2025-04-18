


<br>

![User interface for the Agentic RAG project](readme-images/static/agentic-rag-screen-shot.png "User interface for the Agentic RAG project")

# Agentic RAG - Idea to answers in minutes
#### Improve RAG with an agentic approach that:
- handles relevance and accuracy checks, triggering web search if needed
- does hallucination control on results
- lets you configure prompts for each part of the pipeline

#### Do inference with free API endpoints or self-host on your own GPUs:
- Pre-configured to use free endpoints on build.nvidia.com
- OR, configure to use self-hosted models on your remote GPUs

#### Modify this application and make it your own:
- [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) this repository instead of cloning
- Edit the code to change the Gradio app and logic of agent
 


## Get Started 

#### This app runs in [NVIDIA AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html), a free, lightweight developer platform on your own systems for a cloud-like-UX, no cloud needed.

### Be up and running in 4 minutes (Workbench already installed) 
- Have your [NVIDIA API key](https://org.ngc.nvidia.com/setup/api-keys) on hand
- Have your [Tavily API key](https://tavily.com/) on hand
- Clone this project in Workbench and enter your API keys when prompted
- Click "Open Chat" and add documents and websites for your context
- Start chatting using the API endpoints




*Navigating the README:* [Application Overview](#the-agentic-rag-application) | [Get Started](#get-started) | [Deep Dive](#deep-dive-on-self-hosted-endpoints) | [License](#license)

<!-- Links -->
*Other Resources:* [:arrow_down: Download AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) | [:book: User Guide](https://docs.nvidia.com/ai-workbench/) |[:open_file_folder: Other Projects](https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html) | [:rotating_light: User Forum](https://forums.developer.nvidia.com/t/support-workbench-example-project-agentic-rag/303414)

## Need, Don't Need and Nice to Have

- **Need**: Internet access while running the chat app and two API keys and [NVIDIA API key](https://org.ngc.nvidia.com/setup/api-keys))
- **Don't Need**: Local GPU
- **Nice to Have**: Remote GPU system if you want to self-host a NIM endpoint

## The Agentic RAG Application
#### Using the Application

1. **Clone** the project with AI Workbench, **configure** the relevant API keys, and **start** the chat app
2. **Configure** the separate model components for agent
3. **Add** your documents to the context (vector db) 
4. **Make** a query.
5. **Agent returns** an answer vetted for relevance, accuracy and hallucination. 

 
<img src="./code/chatui/static/agentic-flow.png" width="100%" height="auto">


#### Modifying the Application

* **Within the app you can**:
   * Change the prompts for the different components, e.g. the hallucination grader.
   * Change the webpages and pdfs you want to use for the context in the RAG.
   * Select different endpoints from [build.nvidia.com](https://build.nvidia.com/explore/discover) for the inference components.
   * Configure it to use self-hosted endpoints with [NVIDIA Inference Microservices (NIMs)](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama3-8b-instruct/tags) or [Ollama](https://hub.docker.com/r/ollama/ollama).
* **Within the code you can**:
   * Add new endpoints and endpoint providers
   * Change the Gradio interface or the application structure and logic.

> **Note** Setting up self-hosted endpoints is relatively advanced because you will need to do it manually. 

## Get Started

#### Prerequisites 

1. Install [AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/overview.html).

2. Get an NVIDIA Developer Account and an API key.
   * Go to [build.nvidia.com](https://build.nvidia.com/) and click `Login`.
   * Create account, verify email.
   * Create a Cloud Account.
   * Click your initial > `API Keys`.
   * Create and save your key because you may need it for other projects.

3. Get a Tavily account and an API key.
   * Go to [Tavily](https://tavily.com/) and create an account.
   * Create an API key on the overview page.
     
4. Configure the NVIDIA and Tavily API keys as [secret environment variables in Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/environment/variables.html#basic-usage-for-environment-variables)


> **Note**: NVIDIA Employees: Configure the `INTERNAL_API` environment variable to use endpoints.


### Start the Chat
   
1. Open NVIDIA AI Workbench. Select a [location to work in](https://docs.nvidia.com/ai-workbench/user-guide/latest/locations/locations.html).
   
2. Use the repository URL to clone this project with AI Workbench and wait for it to build.
   
3. Add your NVIDIA API key and the Tavily API key when prompted.

4. Open the **Chat** from Workbench. It should automatically open in a new browser tab.

5. Upload your documents and change the Router prompt to focus on your uploaded documents. 

6. Start chatting.

## Deep Dive on Self-Hosted Endpoints

> **Note** This assumes you've done the "Get Started" steps.

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
