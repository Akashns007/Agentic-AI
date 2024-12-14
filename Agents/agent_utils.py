from crew_ai.all_agents import fetch_latest_news, fetch_internet_search_results
from ollama_utils import process_with_groq

def check_agent_call(text):
    """
    Determine which agent to call based on the query.
    """
    instructions = f"""
    you will act as an intent recognizer.
    Your can only reply with one word and no more than that:
    user query = {text}
    1. If the user query is refering anything about searching on the internet then you will reply with the word = 'internetSearch'
    2. If the user query is asking for news then you will reply with the word = 'newsSearch'
    3. if anything else you will reply with the word = 'NO'
    """
    result = process_with_groq(query=instructions)
    print(result)
    return result

def dispatch_agent(user_query):
    """
    Map the intent to agents and execute the appropriate action.
    """
    intent = check_agent_call(user_query).lower()
    
    if intent == "internetsearch":
        print("Calling Internet Search Agent...")
        return fetch_internet_search_results(user_query)
    
    elif intent == "newssearch":
        print("Calling News Search Agent...")
        return fetch_latest_news(user_query)
    
    elif intent == "no":
        return "Sorry, I cannot process your request. Please rephrase or clarify your query."
    
    else:
        return "Error: Unknown intent returned by the intent recognizer."
