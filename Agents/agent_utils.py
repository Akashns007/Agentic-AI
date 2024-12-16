from crew_ai.all_agents import fetch_latest_news, fetch_internet_search_results,extract_pdf_information,extract_text_information,fetch_youtube_video_data
from ollama_utils import handle_response_stream, process_with_groq, process_with_ollama

def check_agent_call(text):
    """
    Determine which agent to call based on the query.
    """
    instructions = f"""
                You are an intent recognizer. 
                Your *ONLY* output must be a single word, chosen from the list below. Absolutely *NO* other text, explanations, or punctuation is allowed.

                **Input:** User query: {text}

                **Possible Outputs (Choose ONE):**

                1.  **internetSearch:** If the user query refers to general internet searching.
                2.  **newsSearch:** If the user query asks for news.
                3.  **pdfSearch:** If the user query asks about PDF files or searching within them.
                4.  **textSearch:** If the user query asks about text files or searching within them.
                5.  **youtubeSearch:** If the user query asks for videos or learning resources (courses).
                6.  **NO:** For any other type of query.

                **RULES (YOU MUST OBEY THESE):**

                *   **ONE WORD ONLY:** Your entire response *must* consist of a single word selected from the list above.
                *   **NO EXPLANATIONS:** Do not provide any additional text, reasoning, or commentary.
                *   **NO PUNCTUATION:** Do not use periods, commas, or any other punctuation marks.
                *   **CASE SENSITIVE:** Output must match the case of the words in the list (e.g., "internetSearch", not "InternetSearch" or "internetsearch").

                **Example:**

                *   **Input:** "What's the weather in London?"
                *   **Output:** internetSearch

                *   **Input:** "Find me a tutorial on Python."
                *   **Output:** youtubeSearch

                *   **Input:** "Summarize this PDF document."
                *   **Output:** pdfSearch"""
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
        # yt_output = 
        # query = yt_output + "/n this is the result from Youtube api, process and summerize it and response to user that they can refer this video."
        # final_res = process_with_ollama(query)
        return fetch_youtube_video_data(user_query)
        
    elif intent == "no":
        return "Sorry, I cannot process your request. Please rephrase or clarify your query."
    
    else:
        return "Error: Unknown intent returned by the intent recognizer."


if __name__ == "__main__":
    user_query = "best video to learn assembly"
    response = dispatch_agent(user_query)
    print(response)