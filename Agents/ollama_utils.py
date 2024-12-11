import ollama
from speech_utils import speak

def process_with_ollama(model, query):
    """
    Processes the query with the specified Ollama model and returns the streamed response.
    """
    return ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': query}],
        stream=True,
    )

def handle_response_stream(response):
    """
    Handles the response stream from Ollama, speaking the output in chunks.
    """
    paragraph_buffer = ""  # Initialize a buffer to store paragraph text
    
    for chunk in response:
        print(paragraph_buffer)
        if "." in paragraph_buffer:  
            speak(paragraph_buffer)  
            paragraph_buffer = "" 
        
        paragraph_buffer += chunk['message']['content']  # Add content to buffer

    # Speak any remaining text in the buffer
    if paragraph_buffer:
        speak(paragraph_buffer)
        paragraph_buffer = ""
