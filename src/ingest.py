import os
import glob
import json
from sentence_transformers import SentenceTransformer
from client import EndeeClient
import time
from utils import extract_text, chunk_text

def ingest_data():
    # Initialize client
    db_url = os.getenv("ENDEE_DB_URL", "http://localhost:8080")
    client = EndeeClient(base_url=db_url, api_key="secret")
    
    # Wait for server
    max_retries = 5
    for i in range(max_retries):
        try:
            client.health()
            print("Server is healthy")
            break
        except Exception as e:
            print(f"Waiting for server... {e}")
            time.sleep(2)
    else:
        print("Server not reachable")
        return

    # Create Index
    index_name = "demo_docs"
    # Dimensions for all-MiniLM-L6-v2 is 384
    try:
        client.create_index(index_name, dim=384, space_type="l2")
        print(f"Index {index_name} created")
    except Exception as e:
        print(f"Index creation failed (maybe exists): {e}")

    # Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load Data
    # For demo, we'll create some dummy data files if they don't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    dummy_files = {
        "endee_intro.txt": "Endee is a high-performance vector database designed for AI applications.",
        "rag_explained.txt": "RAG stands for Retrieval Augmented Generation, a technique to enhance LLM responses with external data.",
        "python_bindings.txt": "Endee supports Python via REST API and can be easily integrated into data pipelines."
    }
    
    for fname, content in dummy_files.items():
        with open(os.path.join(data_dir, fname), "w") as f:
            f.write(content)

    # Scan for TXT, PDF, and DOCX files
    file_paths = []
    for ext in ["*.txt", "*.pdf", "*.docx"]:
        file_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    
    batch = []
    doc_mapping = {}
    id_counter = 1
    
    for path in file_paths:
        file_name = os.path.basename(path)
        print(f"Processing {file_name}...")
        
        # Extract text from PDF, DOCX, or TXT
        text = extract_text(path)
        if not text.strip():
            print(f"Warning: No text extracted from {file_name}")
            continue
            
        # Segment text into overlapping chunks
        chunks = chunk_text(text, chunk_size=800, overlap=150)
        print(f"Created {len(chunks)} chunks from {file_name}")
        
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            
            item = {
                "id": str(id_counter),
                "vector": embedding,
            }
            batch.append(item)
            
            # Map chunk ID back to chunk content
            doc_mapping[str(id_counter)] = chunk
            id_counter += 1
            
    # Save document mapping for retrieval because Endee doesn't store metadata/text payload yet
    with open("doc_mapping.json", "w") as f:
        json.dump(doc_mapping, f)
            
    if batch:
        print(f"Inserting {len(batch)} vectors...")
        client.insert_vectors(index_name, batch)
        print("Insertion complete")

if __name__ == "__main__":
    ingest_data()
