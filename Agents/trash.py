
import time
import ollama
import sounddevice as sd
import speech_recognition as sr
from TTS.api import TTS
from crewai import Agent, Task
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from MarkdownTools import markdown_validation_tool

load_dotenv()

# Define TTS
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", gpu=True)

# Define wake word and LLM
WAKE_WORD = "nexa"
llm = ChatOpenAI(model="crewai-mistral", base_url="http://localhost:11434/v1")

def speak(text):
    """Convert text to speech."""
    try:
        audio = tts.tts(text, return_type="numpy")
        sd.play(audio, samplerate=22050)
        sd.wait()
    except Exception as e:
        print(f"Error during speech playback: {e}")

def detect_wake_word():
    """Listen for the wake word and return user query."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the wake word...")
        while True:
            try:
                wake_word_audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(wake_word_audio).lower()
                if WAKE_WORD in command:
                    print("Wake word detected! Listening for your query...")
                    recognizer.pause_threshold = 2.0
                    query_audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
                    text = recognizer.recognize_google(query_audio)
                    print("You said: " + text)
                    return text
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")

def process_markdown_document(filename):
    """Validate a Markdown document and return feedback."""
    general_agent = Agent(
        role='Requirements Manager',
        goal="""Provide a detailed list of the markdown 
                linting results. Summarize actionable tasks 
                for the developer.""",
        backstory="""You are an expert QA specialist providing detailed, actionable feedback.""",
        allow_delegation=False, 
        verbose=True,
        tools=[markdown_validation_tool],
        llm=llm
    )

    syntax_review_task = Task(
        description=f"""
            Use the markdown_validation_tool to review the file at this path: {filename}.
            Return a list of changes the developer should make. DO NOT recommend ways to update or modify the document.
        """,
        agent=general_agent
    )
    
    updated_markdown = syntax_review_task.execute()
    return updated_markdown

def handle_query(query):
    """Process user queries."""
    if "validate markdown" in query.lower():
        speak("Please provide the filename to validate.")
        filename = input("Enter the markdown filename: ")
        try:
            result = process_markdown_document(filename)
            print("Validation Results:\n", result)
            speak("The validation is complete. Check your console for details.")
        except Exception as e:
            print(f"Error processing Markdown document: {e}")
            speak("There was an error processing the Markdown document.")
    else:
        response = process_with_ollama('mistral', query)
        handle_response_stream(response)

def process_with_ollama(model, query):
    """Process the query with the specified Ollama model."""
    return ollama.chat(
        model=model,
        messages=[{'role': 'system', 'content': query}],
        stream=True
    )

def handle_response_stream(response):
    """Stream and speak response from Ollama."""
    paragraph_buffer = ""
    for chunk in response:
        paragraph_buffer += chunk['message']['content']
        if "." in paragraph_buffer:
            speak(paragraph_buffer.strip())
            paragraph_buffer = ""

    if paragraph_buffer.strip():
        speak(paragraph_buffer.strip())

def main():
    """Main assistant loop."""
    try:
        while True:
            query = detect_wake_word()
            if query:
                handle_query(query)
    except KeyboardInterrupt:
        print("Assistant stopped by user.")

if __name__ == "__main__":
    main()
