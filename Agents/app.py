import streamlit as st
import os
from agent_utils import dispatch_agent
from ollama_utils import process_with_ollama

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to a temporary directory and return the file path.
    """
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    # Page Title
    st.title("AI Chat Assistant with File Processing")
    st.write("Upload a file and interact with the AI assistant.")

    # Upload File Section
    uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
    if uploaded_file:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write(f"Processing the file in the background...")

    # Chat Interface
    with st.container():
        st.subheader("Chat with the Assistant")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input box for user query
        user_query = st.chat_input("Enter your query here")

        if user_query:
            # Process query
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            if uploaded_file:
                # Inject file into agent processing if required
                response = dispatch_agent(user_query)  # Your logic handles file internally
            else:
                response = "No file uploaded. Please upload a file for file-related queries."

            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

if __name__ == "__main__":
    main()
