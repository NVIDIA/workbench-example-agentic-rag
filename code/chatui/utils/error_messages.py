# chatui/utils/error_messages.py

QUERY_ERROR_MESSAGES = {
    "GraphRecursionError": {
        "title": "⚠️ Too many reasoning steps",
        "body": (
            "The agent tried to answer your question but after several rounds of reasoning it couldn’t make progress.\n\n"
            "This can happen due to a variety of factors related to the query, the documents or the model you are using.\n\n"
            "**Tips:**\n"
            "- Make your question more specific\n"
            "- Use a different or higher precision model\n"
            "- Use different documents or URLs for the context"
        )
    },
    "TavilyAPIError": {
        "title": "⚠️ Tavily Error",
        "body": (
            "The Tavily web search failed.\n\n"
            "**Try:**\n"
            "- Verify your Tavily API key is correct\n"
            "- In the **Workbench Desktop App**, go to **Project Container > Variables**\n"
            "- Re-enter your Tavily API key\n"
            "- Restart project container > Restart chat app > Repeat your query"
        )
    },
    "AuthenticationError": {
        "title": "🚫 API Authentication Error",
        "body": (
            "It looks like your NGC API key is missing or incorrect.\n\n"
            "**Try:**\n"
            "- In the **Workbench Desktop App**, go to **Project Container > Variables**\n"
            "- Re-enter your NVIDIA API key\n"
            "- Restart project container > Restart chat app > Repeat your query"
        )
    },
    "HTTPError": {
        "title": "🔌 Remote API Failure",
        "body": "The remote model service responded with an error. Try again or check your configuration."
    },
    "OutputParserError": {
        "title": "⚠️ Unexpected model output",
        "body": (
            "One of the grader models returned a response that was not the expected JSON format, "
            "so the agent could not finish verifying this answer.\n\n"
            "**Try:**\n"
            "- Submit the query again\n"
            "- Select a different model for the component shown in the Monitor tab\n"
            "- If you customized a prompt, make sure it still asks for a JSON response"
        )
    },
    "Unknown": {
        "title": "❌ Unexpected Error",
        "body": (
            "Something went wrong. Please check the **Chat** logs in the Workbench Desktop App.\n\n"
            "- Go to the **Workbench Desktop App** tab\n"
            "- Click **Output** at the bottom left of the Project tab\n"
            "- Select **Chat** from the dropdown\n"
            "- Check the logs for more details"
        )
    }
}