import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
from typing import List, Tuple, Dict
from io import BytesIO

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from unstructured.partition.pdf import partition_pdf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Constants
COLLECTION_NAME = "documents_collection"
MODEL_NAME = "gpt-4"

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

os.makedirs("/tmp/.chroma", exist_ok=True)

client = chromadb.EphemeralClient()

# Initialize the default embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}
)

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data
def generate_word_cloud(text: str) -> BytesIO:
    """Generate a word cloud image from the given text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer

def create_embeddings(text: str) -> List[float]:
    """Create embeddings for the given text."""
    return default_ef([text])[0]

@st.cache_data
def extract_paragraphs_from_pdf(pdf_file) -> List[Dict[str, str]]:
    """Extract paragraphs from a PDF file."""
    elements = partition_pdf(file=pdf_file)
    return [
        {
            "text": element.text.strip(),
            "page": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
        }
        for element in elements
        if hasattr(element, 'text') and element.text.strip()
    ]

def add_to_collection(paragraph: Dict[str, str], pdf_name: str) -> None:
    """Add a paragraph to the ChromaDB collection."""
    collection.add(
        embeddings=[create_embeddings(paragraph['text'])],
        metadatas=[{"page": paragraph['page'], "pdf_name": pdf_name}],
        documents=[paragraph['text']],
        ids=[f"{pdf_name}_page_{paragraph['page']}_{collection.count()}"]
    )

def search_keywords(query: str, k: int = 10) -> List[Tuple[str, Dict, float]]:
    """Search for keywords in the collection."""
    results = collection.query(
        query_embeddings=[create_embeddings(query)],
        n_results=k,
        include=['metadatas', 'documents', 'distances']
    )
    return list(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]))

def generate_prompt(query: str, context_chunks: List[Tuple[str, Dict]]) -> str:
    """Generate a prompt for the OpenAI API."""
    prompt = "Context information is below.\n---------------------\n"
    prompt += "\n".join(f"{i}. {chunk}" for i, (chunk, _) in enumerate(context_chunks, 1))
    prompt += f"\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer: "
    return prompt

def chat_with_document(user_input: str) -> str:
    """Chat with the document using OpenAI API."""
    all_documents = collection.get()['documents']
    full_context = " ".join(all_documents)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Here's the context:\n{full_context}"},
        {"role": "assistant", "content": "OK, I understand the context. How can I help you?"}
    ]
    
    messages.extend(st.session_state.chat_history)
    messages.append({"role": "user", "content": user_input})
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    
    assistant_message = response.choices[0].message.content
    
    new_chat_history = st.session_state.chat_history.copy()
    new_chat_history.append({"role": "user", "content": user_input})
    new_chat_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message, new_chat_history

@st.cache_data
def generate_csv(results: Dict[str, List[Tuple[str, Dict, float]]]) -> str:
    """Generate a CSV file from search results."""
    data = [
        {
            'pdf_name': metadata.get('pdf_name', ''),
            'keyword': keyword,
            'page': metadata.get('page', ''),
            'text': text,
            'distance': distance
        }
        for keyword, keyword_results in results.items()
        for text, metadata, distance in keyword_results
    ]
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def main():
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = set()
        st.session_state.pdf_uploaded = False
        st.session_state.word_cloud_image = None
        st.session_state.chat_history = []

    st.sidebar.title("Settings")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    col1, col2 = st.columns(2)

    with col1:
        st.title("Document Analysis Tool")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            pdf_name = uploaded_file.name
            
            if pdf_name not in st.session_state.processed_pdfs:
                paragraphs = extract_paragraphs_from_pdf(uploaded_file)

                progress_bar = st.progress(0)
                status_text = st.empty()

                full_text = ""
                for i, paragraph in enumerate(paragraphs):
                    add_to_collection(paragraph, pdf_name)
                    full_text += paragraph['text'] + " "
                    progress = (i + 1) / len(paragraphs)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing paragraph {i+1} of {len(paragraphs)}")

                progress_bar.empty()
                status_text.empty()

                st.session_state.processed_pdfs.add(pdf_name)
                st.session_state.pdf_uploaded = True
                st.session_state.word_cloud_image = generate_word_cloud(full_text)

                st.success(f"PDF '{pdf_name}' uploaded and processed successfully!")

        if st.session_state.pdf_uploaded:
            st.image(st.session_state.word_cloud_image, caption='Word Cloud of the Document', use_column_width=True)

            keywords = st.text_input("Enter keywords (comma-separated)")

            if st.button("Search"):
                if keywords:
                    keyword_list = [k.strip() for k in keywords.split(",")]
                    
                    with st.spinner("Searching..."):
                        results = {keyword: search_keywords(keyword) for keyword in keyword_list}

                    table_data = []
                    for keyword, keyword_results in results.items():
                        for i, (text, metadata, distance) in enumerate(keyword_results):
                            table_data.append({
                                "Result #": i+1,
                                "Keyword": keyword,
                                "Page": metadata.get('page', 'N/A'),
                                "Text": text,
                                "Distance": distance
                            })
                    
                    st.dataframe(table_data)
                    st.write("---")
                else:
                    st.warning("Please enter keywords to search.")

    with col2:
        col2_1, col2_2 = st.columns([4, 1])
        with col2_1:
            st.title("Chat")
        with col2_2:
            if st.button("Clear"):
                st.session_state.chat_history = []
        
        if st.session_state.pdf_uploaded:
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_history:
                    st.chat_message(message["role"]).write(message["content"])
            
            user_input = st.chat_input("Ask a question about the document:")
            
            if user_input:
                if openai_api_key:
                    with chat_container:
                        st.chat_message("user").write(user_input)
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response, new_chat_history = chat_with_document(user_input)
                            st.write(response)
                    
                    st.session_state.chat_history = new_chat_history
                else:
                    st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            st.info("Please upload a document first to enable chat functionality.")

if __name__ == "__main__":
    main()
