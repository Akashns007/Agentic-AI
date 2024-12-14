import os
import sys
from crewai import Agent, Crew, Task
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai_tools import CSVSearchTool

load_dotenv()

# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="groq/mixtral-8x7b-32768"
# )
    
llm = ChatOpenAI(
    model="ollama/mistral",
    base_url="http://localhost:11434"
)

csv_search_agent = Agent(
    role="PDF Search Agent",
    goal="""
        Act as a document retrieval and refinement agent. Take raw csv files 
        as input and extract relevant information based on the user's query.
        Ensure the output is accurate, concise, and directly aligned 
        with the query.
    """,
    backstory="""
        You are a highly trained document analysis agent specializing in 
        retrieving and refining information from large csv files. Your 
        expertise lies in identifying the most relevant details based on 
        user queries and presenting them clearly and accurately.
    """,
    verbose=True,
    llm=llm,  # Use the same LLM configuration
)




# Initialize the TXTSearchTool
search_tool = CSVSearchTool(
    csv = sys.argv[1],
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mixtral-8x7b-32768",
                temperature=0.1,
                top_p=1,
                stream=True,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                # title="Embeddings",
            ),
        ),
    )
)


# Define the task
csv_retrieval_task = Task(
    description="""
        Use the CSVSearchTool to process the raw csv files and extract information 
        relevant to the question {question}. Ensure the output is concise, accurate, and 
        contextually aligned with the query.
    """,
    expected_output="""
        Refined text containing only the information relevant to the question.
    """,
    agent=csv_search_agent,
    tools=[search_tool],  # Use PDFSearchTool instead of TXTSearchTool
)
    
    
crew = Crew(
    agents=[csv_search_agent],
    tasks=[csv_retrieval_task],
    verbose=True,
)

inputs = {"question":"name all the species in the iris dataset"}

result = crew.kickoff(inputs=inputs)
print(result)