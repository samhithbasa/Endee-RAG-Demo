import streamlit as st
from client import EndeeClient
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from groq import Groq

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Endee RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        color: #ffffff;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .result-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .result-score {
        font-size: 0.8rem;
        color: #8b949e;
        margin-bottom: 0.5rem;
    }
    .result-text {
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Fix Chat Message Text Color */
    div[data-testid="stChatMessage"] p {
        color: #ffffff !important;
    }
    div[data-testid="stChatMessage"] .stMarkdown {
        color: #ffffff !important;
    }
    .stMarkdown p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_resources():
    client = EndeeClient(api_key="secret")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return client, model

client, model = get_resources()

# Load doc mapping
DOC_MAPPING_FILE = "doc_mapping.json"
if os.path.exists(DOC_MAPPING_FILE):
    with open(DOC_MAPPING_FILE, "r") as f:
        doc_mapping = json.load(f)
else:
    doc_mapping = {}

def save_doc_mapping(mapping):
    with open(DOC_MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

# Configuration
API_KEYS = {
    "OpenAI": os.getenv("OPENAI_API_KEY", ""),
    "Google Gemini": os.getenv("GEMINI_API_KEY", ""),
    "Groq": os.getenv("GROQ_API_KEY", "")
}

# Sidebar Configuration
with st.sidebar:
    st.title("🤖 Configuration")
    st.markdown("---")
    
    # Status
    try:
        if client.health():
            st.success("Endee DB: Connected")
    except:
        st.error("Endee DB: Disconnected")
    
    st.markdown("### LLM Settings")
    llm_provider = st.selectbox("Provider", ["None (Search Only)", "OpenAI", "Google Gemini", "Groq"])
    
    api_key = API_KEYS.get(llm_provider, "")
    
    if llm_provider != "None (Search Only)" and not api_key:
        api_key = st.text_input("API Key", type="password", placeholder=f"Enter {llm_provider} Key")
    elif llm_provider != "None (Search Only)":
        st.success(f"Using configured key for {llm_provider}")
    
    st.markdown("---")
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Ingest Uploaded Files"):
            with st.spinner("Ingesting files..."):
                batch = []
                # Determine new ID start
                existing_ids = [int(k) for k in doc_mapping.keys() if k.isdigit()]
                next_id = max(existing_ids) + 1 if existing_ids else 1
                
                count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # Read file
                        text_content = uploaded_file.read().decode("utf-8")
                        
                        # Chunking Strategy: Split by double newlines (paragraphs)
                        chunks = [c.strip() for c in text_content.split('\n\n') if c.strip()]
                        
                        for chunk in chunks:
                            # Embed
                            vector = model.encode(chunk).tolist()
                            
                            # Create Item
                            doc_id = str(next_id)
                            batch.append({
                                "id": doc_id,
                                "vector": vector
                            })
                            
                            # Update mapping with CHUNK content
                            doc_mapping[doc_id] = chunk
                            
                            next_id += 1
                            count += 1
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if batch:
                    try:
                        # Ensure index exists (idempotent usually, or catch error)
                        try:
                            client.create_index("demo_docs", dim=384)
                        except:
                            pass # Assume exists
                            
                        # Insert
                        success = client.insert_vectors("demo_docs", batch)
                        if success:
                            save_doc_mapping(doc_mapping)
                            st.success(f"Successfully ingested {count} chunks from uploaded files!")
                        else:
                            st.error("Failed to insert vectors into Endee.")
                    except Exception as e:
                        st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    st.markdown("### Index Info")
    st.info(f"Documents Indexed: {len(doc_mapping)}")

# Main Interface
st.title("Endee RAG Demo")
st.markdown("Ask questions about your documents using **Endee Vector Database**.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate RAG response
def generate_rag_response(query, context, provider, key):
    prompt_template = f"""
    You are a helpful assistant. Use the following context to answer the user's question.
    If the answer is not in the context, say you don't know.
    
    Context:
    {context}
    
    Question: 
    {query}
    
    Answer:
    """
    
    try:
        if provider == "OpenAI" and key:
            openai.api_key = key
            client = openai.OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_template}]
            )
            return response.choices[0].message.content
            
        elif provider == "Google Gemini" and key:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt_template)
            return response.text
            
        elif provider == "Groq" and key:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_template}]
            )
            return response.choices[0].message.content
            
        else:
            return "Please provide a valid API Key to generate an answer."
            
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Query
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Searching Endee..."):
            # 1. Embed Query
            vector = model.encode(prompt).tolist()
            
            # 2. Search Endee
            try:
                # Search
                results = client.search("demo_docs", vector, k=5)
                
                # Parse Results
                matches = []
                if isinstance(results, list):
                    matches = results
                elif isinstance(results, dict) and 'matches' in results:
                    matches = results['matches']
                
                retrieved_docs = []
                
                for match in matches:
                     if isinstance(match, list):
                         doc_id = match[1] # ID is index 1
                         distance = match[0]
                     elif isinstance(match, dict):
                         doc_id = match.get('id') or match.get(b'id')
                         distance = match.get('distance')
                     else:
                         continue
                         
                     if isinstance(doc_id, bytes): doc_id = doc_id.decode()
                     
                     text = doc_mapping.get(str(doc_id), "")
                     if text and text not in retrieved_docs:
                         retrieved_docs.append(text)
                
                # 3. Generate Answer (RAG)
                context = "\n\n".join(retrieved_docs)
                
                if llm_provider != "None (Search Only)" and api_key:
                    message_placeholder.markdown("Generating answer...")
                    prompt_plus = f"The user asked: {prompt}. If they made a typo like 'background' for 'backend', please use your judgment to answer correctly based on the context."
                    full_response = generate_rag_response(prompt_plus, context, llm_provider, api_key)
                else:
                    full_response = "**Retrieved Context:**\n\n" + context + "\n\n*(Connect an LLM in settings to generate an authenticated answer)*"
                
                message_placeholder.markdown(full_response)
                
                # Show Sources
                with st.expander("View Retrieved Sources"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-score">Source {i+1}</div>
                            <div class="result-text">{doc}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {e}")
