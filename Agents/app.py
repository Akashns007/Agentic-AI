import gradio as gr
import time
from speech_utils import detect_wake_word, speak
from agent_utils import dispatch_agent
from ollama_utils import process_with_ollama, handle_response_stream
from timing_utils import calculate_times


def process_query_stream(query, use_voice):
    """
    Processes the query and yields responses in chunks.
    """
    start_time = time.time()

    # Handle query via voice or manual input
    if use_voice and not query:
        query = detect_wake_word()
        if not query:
            yield "Listening for wake word..."
            return

    if query:
        # Simulating agent dispatch logic
        response = dispatch_agent(query)
        if "Sorry" in response or "Error" in response:
            response_stream = process_with_ollama('mistral', query)
            accumulated_response = ""  # Buffer to collect all chunks

            for partial_response in handle_response_stream(response_stream):
                accumulated_response += partial_response  # Append each chunk
                yield accumulated_response  # Yield growing response
        else:
            yield response
            speak(response)  # Optional TTS

        calculate_times(start_time, time.time(), time.time(), time.time())
        return

    yield "No input received."


# Gradio App with Chat-Like Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 NEXUS AI Assistant - Voice & Text Interface")

    # Components
    use_voice = gr.Checkbox(label="Enable Voice Mode", value=False)
    chatbot = gr.Chatbot(label="Conversation")
    query_box = gr.Textbox(label="Enter your query or enable Voice Mode.", lines=1, placeholder="Type your query here...")
    submit_button = gr.Button("Submit")

    # Function to handle the chat logic
    def chatbot_response(chat_history, user_input, use_voice):
        """
        Update the chat history as responses are streamed.
        """
        # Handle voice input if enabled
        if use_voice and not user_input:
            user_input = detect_wake_word()

        if not user_input:
            return chat_history + [("System", "No input received.")], ""

        # Add user input to the chat history (display on the right)
        chat_history.append(("User", user_input))

        # Stream assistant responses
        accumulated_response = ""
        chat_history.append(("Assistant", ""))  # Placeholder for the assistant's response

        for response in process_query_stream(user_input, use_voice):
            accumulated_response = response
            # Update assistant's response in real time (display on the left)
            chat_history[-1] = ("Assistant", accumulated_response)
            yield chat_history, ""

    # Button click behavior
    submit_button.click(
        fn=chatbot_response,
        inputs=[chatbot, query_box, use_voice],
        outputs=[chatbot, query_box]
    )

demo.launch()
