import pandas as pd
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="ollama/mistral",
    base_url="http://localhost:11434"
)

# Initialize SentenceTransformer for semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load CSV file
def load_csv(file_path):
    """Load CSV into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Preprocess CSV
def preprocess_csv(df):
    """Prepare the CSV content for search."""
    df["combined"] = df.apply(lambda row: " | ".join(row.values.astype(str)), axis=1)
    embeddings = model.encode(df["combined"].tolist(), convert_to_tensor=True)
    return df, embeddings

# Perform Retrieval
def retrieve_data(query, df, embeddings, top_k=3):
    """Retrieve relevant rows from the CSV."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    retrieved_indices = [item["corpus_id"] for item in scores[0]]
    return df.iloc[retrieved_indices]

# RAG Workflow
def rag_csv(file_path, query, top_k=3):
    """End-to-end RAG for CSV."""
    df = load_csv(file_path)
    if df is None:
        return "Error: Unable to load CSV file."
    
    df, embeddings = preprocess_csv(df)
    retrieved_rows = retrieve_data(query, df, embeddings, top_k)
    
    # Combine retrieved rows for LLM input
    context = "\n".join(retrieved_rows["combined"].tolist())
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Generate response with LLM
    response = llm(prompt)
    return response

# Example Usage
if __name__ == "__main__":
    file_path = "iris.csv"
    query = "how many species are ther in this dataset"
    result = rag_csv(file_path, query)
    print("\nResponse from RAG:\n", result)
