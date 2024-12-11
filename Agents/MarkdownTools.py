import os
import mistune
from langchain.tools import tool

@tool("markdown_validation_tool")
def markdown_validation_tool(file_path: str) -> str:
    """
    A tool to review files for markdown syntax validity.

    Parameters:
    - file_path: The path to the markdown file to be reviewed.

    Returns:
    - validation_results: A list of potential syntax issues.
    """
    if not os.path.exists(file_path):
        return "Could not validate file. The provided file path does not exist."

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Use Mistune to parse Markdown
        parser = mistune.create_markdown()
        html_output = parser(content)  # Convert to HTML
        if html_output:
            return "Markdown syntax appears valid."
        else:
            return "The Markdown file might be empty or invalid."
    except Exception as e:
        return f"Error during validation: {str(e)}"
