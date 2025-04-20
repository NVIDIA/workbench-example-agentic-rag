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
    "AuthenticationError": {
        "title": "🚫 API Authentication Error",
        "body": (
            "It looks like one of your API keys is missing or incorrect.\n\n"
            "**Fix it:**\n"
            "- Go to the **Workbench Desktop App** tab\n"
            "- Go to **Project Container > Variables**\n"
            "- Re-enter your NVIDIA and Tavily API keys"
            "- Check if you're using a hosted model\n"
            "- Make sure your API key is valid and entered correctly"
        )
    },
    "HTTPError": {
        "title": "🔌 Remote API Failure",
        "body": "The remote model service responded with an error. Try again or check your configuration."
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