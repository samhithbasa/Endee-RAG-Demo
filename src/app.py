import streamlit as st
from client import EndeeClient
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils import extract_text, chunk_text

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Endee RAG - Gemini Powered",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Modern UI
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0d0f14;
        color: #e6edf3;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121620;
        border-right: 1px solid #1f2430;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #090c10;
        color: #ffffff;
        border: 1px solid #2f363d;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 1px #4f46e5;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Premium Result Cards */
    .result-card {
        background-color: #151a26;
        border: 1px solid #232a3b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .result-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    .result-score {
        font-size: 0.8rem;
        font-weight: 600;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .result-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #c9d1d9;
    }
    
    /* Fix Chat Message Typography */
    div[data-testid="stChatMessage"] {
        background-color: #121620;
        border: 1px solid #1f2430;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stChatMessage"] p {
        color: #f0f6fc !important;
        font-family: 'Outfit', sans-serif;
        line-height: 1.6;
    }
    .stMarkdown p {
        color: #f0f6fc !important;
    }
    
    /* Custom Info banner styling */
    .stInfo {
        background-color: #1a1b26;
        border-left: 4px solid #4f46e5;
        color: #a9b1d6;
    }
</style>
""", unsafe_allow_html=True)

# State initialization
if "gemini_key_index" not in st.session_state:
    st.session_state.gemini_key_index = 0

@st.cache_resource
def get_resources():
    db_url = os.getenv("ENDEE_DB_URL", "http://localhost:8080")
    client = EndeeClient(base_url=db_url, api_key="secret")
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

# Helper function to parse comma-separated keys
def parse_keys(input_str):
    if not input_str:
        return []
    return [k.strip() for k in input_str.replace(";", ",").split(",") if k.strip()]

# Loaded keys from environment
env_keys_str = os.getenv("GEMINI_API_KEY", "")
env_keys = parse_keys(env_keys_str)

# Sidebar Configuration
with st.sidebar:
    st.title("🤖 Config Panel")
    st.markdown("---")
    
    # Status Indicators
    db_ok = False
    try:
        if client.health():
            st.success("🟢 Endee DB: Connected")
            db_ok = True
    except:
        st.error("🔴 Endee DB: Disconnected")
    
    st.markdown("### Gemini Status")
    
    # Resolve loaded keys exclusively from the secure .env file
    active_keys = env_keys
    st.session_state.gemini_keys_pool = active_keys
    
    if active_keys:
        st.success("🟢 Google Gemini: Active")
        st.info(f"✨ Gemini Model Loaded ({len(active_keys)} key(s) in rotation)")
    else:
        st.warning("⚠️ No Gemini keys configured in .env. Retrieval only active.")
        
    st.markdown("---")
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader("Upload files (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
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
                        file_bytes = uploaded_file.read()
                        
                        # Use unified text extractor
                        text_content = extract_text(file_bytes, uploaded_file.name)
                        if not text_content.strip():
                            st.warning(f"No text content could be extracted from {uploaded_file.name}")
                            continue
                        
                        # Use smart overlapping chunking
                        chunks = chunk_text(text_content, chunk_size=800, overlap=150)
                        
                        for chunk in chunks:
                            vector = model.encode(chunk).tolist()
                            doc_id = str(next_id)
                            batch.append({
                                "id": doc_id,
                                "vector": vector
                            })
                            doc_mapping[doc_id] = chunk
                            
                            next_id += 1
                            count += 1
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if batch:
                    try:
                        try:
                            client.create_index("demo_docs", dim=384)
                        except:
                            pass # Assume index already exists
                            
                        success = client.insert_vectors("demo_docs", batch)
                        if success:
                            save_doc_mapping(doc_mapping)
                            st.success(f"Successfully ingested {count} chunks!")
                        else:
                            st.error("Failed to insert vectors into Endee.")
                    except Exception as e:
                        st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    st.markdown("### Index Info")
    st.info(f"Passages Indexed: {len(doc_mapping)}")

# Main Interface
st.title("Endee RAG Demo")
st.markdown("Interact with your documents using **Endee Vector Database** and **Google Gemini**.")

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Rotate and retrieve keys helper
def get_next_gemini_key():
    keys = st.session_state.get("gemini_keys_pool", [])
    if not keys:
        return None
    idx = st.session_state.gemini_key_index % len(keys)
    st.session_state.gemini_key_index += 1
    return keys[idx]

# API call failover runner
def call_gemini_with_failover(func, *args, **kwargs):
    keys = st.session_state.get("gemini_keys_pool", [])
    if not keys:
        raise ValueError("No Gemini keys configured.")
        
    attempts = len(keys)
    last_exception = None
    
    for _ in range(attempts):
        key = get_next_gemini_key()
        try:
            genai.configure(api_key=key)
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # Console log for background debug
            print(f"Failover key trigger. Key failed. Attempting next. Error: {e}")
            
    raise last_exception if last_exception else RuntimeError("All configured Gemini API keys failed.")

# Smart Query Rewriter (Gemini Only)
def condense_query(chat_history, latest_query):
    keys = st.session_state.get("gemini_keys_pool", [])
    if not keys or not chat_history:
        return latest_query
        
    formatted_history = ""
    for msg in chat_history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
        
    prompt = f"""
    Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be searched in a vector database.
    DO NOT answer the question. Just return the rephrased standalone question and nothing else.
    
    Conversation History:
    {formatted_history}
    
    Follow-up Question: {latest_query}
    Standalone Question:
    """
    
    def _api_call():
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    try:
        return call_gemini_with_failover(_api_call)
    except Exception as e:
        print(f"Query condensation failed: {e}")
        return latest_query

# RAG Synthesizer (Gemini Only)
def generate_rag_response(query, context, chat_history):
    keys = st.session_state.get("gemini_keys_pool", [])
    if not keys:
        return "Please configure one or more Gemini API Keys in the sidebar to generate responses."
        
    formatted_history = ""
    for msg in chat_history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"

    prompt_template = f"""
    You are a helpful assistant. Use the following context and conversation history to answer the user's question.
    If the answer is not in the context, say you don't know. Keep your answer concise and accurate.
    
    Context:
    {context}
    
    Conversation History:
    {formatted_history}
    
    Question: 
    {query}
    
    Answer:
    """
    
    def _api_call():
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt_template)
        return response.text
        
    try:
        return call_gemini_with_failover(_api_call)
    except Exception as e:
        return f"Error generating answer after trying all available Gemini keys: {str(e)}"

# Chat Input Handler
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. Standalone Query Condensation
        history_to_pass = st.session_state.messages[:-1]
        search_query = condense_query(history_to_pass, prompt)
        
        if search_query != prompt:
            st.info(f"🔍 Context Rewriting Active. Searching database for standalone query: *\"{search_query}\"*")
        
        with st.spinner("Searching database..."):
            vector = model.encode(search_query).tolist()
            
            try:
                # 2. Search Endee Database
                results = client.search("demo_docs", vector, k=5)
                
                # Parse search results
                matches = []
                if isinstance(results, list):
                    matches = results
                elif isinstance(results, dict) and 'matches' in results:
                    matches = results['matches']
                
                retrieved_docs = []
                
                for match in matches:
                     if isinstance(match, list):
                         doc_id = match[1]
                     elif isinstance(match, dict):
                         doc_id = match.get('id') or match.get(b'id')
                     else:
                         continue
                         
                     if isinstance(doc_id, bytes): 
                         doc_id = doc_id.decode()
                     
                     text = doc_mapping.get(str(doc_id), "")
                     if text and text not in retrieved_docs:
                         retrieved_docs.append(text)
                
                # 3. Formulate Context
                context = "\n\n".join(retrieved_docs)
                
                if st.session_state.get("gemini_keys_pool"):
                    message_placeholder.markdown("Generating answer...")
                    prompt_plus = f"The user asked: {prompt}. If they made a typo like 'background' for 'backend', please use your judgment to answer correctly based on the context."
                    full_response = generate_rag_response(prompt_plus, context, history_to_pass)
                else:
                    full_response = "**Retrieved Context:**\n\n" + context + "\n\n*(Please add a Gemini API Key in the sidebar to generate response answers)*"
                
                message_placeholder.markdown(full_response)
                
                # Show Sources Dropdown
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
