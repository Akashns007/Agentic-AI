import time
from speech_utils import detect_wake_word
from ollama_utils import process_with_ollama, handle_response_stream
from timing_utils import calculate_times

def main():
    """
    Main function to handle the entire speech-to-speech process.
    """
    while True:
        # Step 1: Wait for wake word
        start_time = time.time()

        # Step 2: Listen to query
        query = detect_wake_word()

        # Step 3: Process query with Ollama
        response = process_with_ollama('mistral', query)
        ollama_response_time = time.time()

        # Step 4: Handle streamed response and speak it
        handle_response_stream(response)
        speech_generation_time = time.time()

        # Step 5: Calculate and print timings
        end_time = time.time()
        calculate_times(start_time, ollama_response_time, speech_generation_time, end_time)

# Run the main function
if __name__ == "__main__":
    main()
