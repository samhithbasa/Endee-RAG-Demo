import os
import glob
from sentence_transformers import SentenceTransformer
from client import EndeeClient
import time

def ingest_data():
    # Initialize client
    client = EndeeClient(api_key="secret")
    
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

    file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    
    batch = []
    id_counter = 1
    
    for path in file_paths:
        with open(path, "r") as f:
            text = f.read()
            
        # Simple chunking by sentence or just whole file for short texts
        chunks = [text] # simplified
        
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            
            item = {
                "id": str(id_counter),
                "vector": embedding,
                # "text": chunk # Endee might not store payload in vector/insert? 
                # According to code, it only stores vector, sparse, id. 
                # We need a separate mapping or rely on ID to retrieve text.
                # For this demo, let's assume we fetch text from file params or just map ID back to text in memory for simplicity.
            }
            # We'll store text in a separate map for the demo app to retrieve
            # For a real app, use a DB or Endee might support metadata in future? 
            # Checked source: HybridVectorObject has id, vector, sparse... no metadata field seen in `parse_obj`.
            
            batch.append(item)
            id_counter += 1
            
    # Save document mapping for retrieval because Endee doesn't store metadata/text payload yet
    import json
    doc_mapping = {}
    current_id = 1
    for path in file_paths:
        with open(path, "r") as f:
            doc_mapping[str(current_id)] = f.read()
            current_id += 1
            
    with open("doc_mapping.json", "w") as f:
        json.dump(doc_mapping, f)
            
    if batch:

        print(f"Inserting {len(batch)} vectors...")
        client.insert_vectors(index_name, batch)
        print("Insertion complete")

if __name__ == "__main__":
    ingest_data()
