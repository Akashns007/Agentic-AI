import os
from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

def fetch_internet_search_results(topic):
    """
    Function to perform an internet search for a given topic using CrewAI and SerperDevTool.
    Returns a summarized result of the query along with relevant links.

    Args:
        topic (str): The search query or topic for the internet search.

    Returns:
        str: A summarized report of the search results.
    """
    # Load environment variables
    load_dotenv()

    # Define the LLM
    llm = ChatOpenAI(
        model="ollama/mistral",
        base_url="http://localhost:11434"
    )

    # Define the Internet Search Agent
    internet_search_agent = Agent(
        role="Internet Search Agent",
        goal=""" 
            Your task is to find information on the internet using the 'Search the internet' tool. 
            Use it to execute a search query based on the user's input, evaluate the credibility 
            of the returned results, and provide a clear summary of the information.
        """,
        backstory=""" 
            You are a specialized agent for web searches, extracting reliable and concise information 
            from the internet to address user queries.
        """,
        verbose=True,
        llm=llm,
    )

    # Initialize the InternetSearchTool
    search_tool = SerperDevTool(
        n_results=2,
    )

    # Define the Task
    internet_search_task = Task(
        description=f""" 
            Use the 'search_tool' tool to find information about the topic '{topic}'.
            You should:
            1. Review the results returned by the tool.
            2. Summarize the findings along with the links in a concise and clear manner.
        """,
        expected_output=""" 
            A summarized report of relevant information gathered about the topic.
        """,
        agent=internet_search_agent,
        tools=[search_tool],
    )

    # Crew Definition
    crew = Crew(
        agents=[internet_search_agent],
        tasks=[internet_search_task],
        verbose=True,
    )

    # Execute the Task
    result = crew.kickoff(inputs={"topic": topic})
    return result
