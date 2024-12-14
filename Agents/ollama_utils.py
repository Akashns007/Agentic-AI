import os
from langchain_groq import ChatGroq
import ollama
from speech_utils import speak

def process_with_ollama(model="mistral", query=''):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    """
    return ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': query}],
        stream=True,
    )
    
def process_with_groq(query=''):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    """
    llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768"
    )
    message =  [
        (
            "system",
            "You are an intent recognition model",
        ),
        ("human", query),
    ]
    intent = llm.invoke(message)
    
    return intent.content

    
def handle_response_stream(response):
    """
    Handles the response stream from Ollama, speaking the output in chunks.
    """
    
    paragraph_buffer = ""  # Initialize a buffer to store paragraph text
    
    for chunk in response:
        
        if "." in paragraph_buffer and len(paragraph_buffer) > 50:  
            print(paragraph_buffer)
            speak(paragraph_buffer)  
            paragraph_buffer = "" 
        
        paragraph_buffer += chunk['message']['content']  # Add content to buffer

    # Speak any remaining text in the buffer
    if paragraph_buffer:
        print(paragraph_buffer)
        speak(paragraph_buffer)
        paragraph_buffer = ""


if __name__ == '__main__':
    res = process_with_groq(query="""you will act as an intent recognizer.
    Your can only reply with one word and no more than that:
    user query = "look up on the internet for 5 best phones" 
    1. If the user query is refering anything about searching on the internet then you will reply with the word = 'internetSearch'
    2. If the user query is asking for news then you will reply with the word = 'newsSearch'
    3. if anything else you will reply with the word = 'NO'
    """
    )
    print(res)