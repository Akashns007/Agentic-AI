from crewai_tools import PDFSearchTool
import sys

tool = PDFSearchTool(
    pdf=sys.argv[1],
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistral:latest",
                temperature=0.2,
                top_p=0.9,
                stream=True,
            ),
        ),
        embedder=dict(
            provider="ollama", # or openai, ollama, ...
            config=dict(
                model="nomic-embed-text:latest"
                
            ),
        ),
    )
)

answer = tool.run(sys.argv[2])
print(answer)