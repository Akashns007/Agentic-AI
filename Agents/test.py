import time
import threading
from speech_utils import detect_wake_word, speak
from agent_utils import dispatch_agent
from ollama_utils import process_with_groq, process_with_ollama, handle_response_stream
from timing_utils import calculate_times

def handle_agent_response(query):
    """
    Handles the agent's response in a separate thread.
    Allows Mistral to respond first, and agents to work in the background.
    """
    response = dispatch_agent(query)
    if response and "Sorry" not in response and "Error" not in response:
        print("Agent Response:", response)
        speak(response)  # Speak the agent's response
    else:
        print("Agent found no valid response or encountered an error.")

def main():
    """
    Main function for fast conversational interaction with Mistral,
    while agents process queries in the background.
    """
    
    
    while True:
        try:
            # Step 1: Detect wake word and listen to user query
            start_time = time.time()
            query = detect_wake_word()

            if query:

                # Step 2: Immediately process with Mistral (quick response)
                print("Processing with Mistral...")
                response = process_with_ollama('mistral', query)
                handle_response_stream(response)  # Stream Mistral's response quickly

                # Step 3: Dispatch agents in parallel (background processing)
                agent_thread = threading.Thread(target=handle_agent_response, args=(query,))
                agent_thread.start()

                # Step 4: Track timings
                end_time = time.time()
                calculate_times(start_time, time.time(), time.time(), end_time)

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break

if __name__ == "__main__":
    main()