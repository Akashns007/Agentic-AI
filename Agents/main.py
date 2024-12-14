import time
from speech_utils import detect_wake_word, speak
from agent_utils import dispatch_agent
from ollama_utils import process_with_ollama, handle_response_stream
from timing_utils import calculate_times

def main():
    """
    Main function to handle the entire speech-to-speech process.
    """
    while True:
        try:
            # Step 1: Wait for wake word and listen to query
            start_time = time.time()
            
            query = detect_wake_word()
            
            if query:
                # Step 2: Determine and dispatch agent
                print("Processing query with intent recognition...")
                response = dispatch_agent(query)

                if "Sorry" in response or "Error" in response:
                    # If no valid agent matched, fallback to Ollama processing
                    print("Fallback to general query processing...")
                    response = process_with_ollama('mistral', query)
                    handle_response_stream(response)
                else:
                    print("Agent Response: ", response)
                    speak(response)  # Speak the agent's response

                # Step 3: Timings
                end_time = time.time()
                calculate_times(start_time, time.time(), time.time(), end_time)

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break

if __name__ == "__main__":
    main()