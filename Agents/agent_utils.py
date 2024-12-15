from crew_ai.all_agents import fetch_latest_news, fetch_internet_search_results,extract_pdf_information,extract_text_information,fetch_youtube_video_data
from ollama_utils import process_with_groq

def check_agent_call(text):
    """
    Determine which agent to call based on the query.
    """
    instructions = f"""
    you will act as an intent recognizer.
    Your can only reply with one word and no more than that:
    below is the user query...based on the query select one of the mentioned words below and reply only that word
    user query = {text}
    1. If the user query is refering anything about searching on the internet then you will reply with the word = 'internetSearch'
    2. If the user query is asking for news then you will reply with the word = 'newsSearch'
    3. If the user query is asking for any information about a pdf file or to search something in a pdf file then you will reply with the word = 'pdfSearch'
    4. If the user query is asking for any information about a text file or search something in a text file then you will reply with the word = 'textSearch'
    5. If the user query is asking for any kind of videos, or courses to learn anything then you will reply with the word = 'youtubeSearch'
    6. if anything else you will reply with the word = 'NO'
    
    Rule: You can only reply with those specified words and NOTHING other that.
    
    You must follow all the rules...

    """
    message =  [
        (
            "system",
            "You are an intent recognition model",
        ),
        ("human", instructions),
    ]
    result = process_with_groq(query=message)
    print(result)
    return result

def dispatch_agent(user_query):
    """
    Map the intent to agents and execute the appropriate action.
    """
    intent = check_agent_call(user_query).lower()
    
    if "internetsearch" in intent:
        print("Calling Internet Search Agent...")
        return fetch_internet_search_results(user_query)
    
    elif "newssearch" in intent:
        print("Calling News Search Agent...")
        return fetch_latest_news(user_query)
    
    elif "pdfsearch" in intent:
        print("Calling PDF Search Agent...")
        return extract_pdf_information()
    
    elif "textsearch" in intent:
        print("Calling Text Search Agent...")
        return extract_text_information()
    
    elif "youtubesearch" in intent:
        print("Calling YouTube Search Agent...")
        return fetch_youtube_video_data(user_query)
    
    elif intent == "no":
        return "Sorry, I cannot process your request. Please rephrase or clarify your query."
    
    else:
        return "Error: Unknown intent returned by the intent recognizer."
