# Endee RAG Demo

This project demonstrates a Semantic Search / RAG system using the [Endee Vector Database](https://github.com/EndeeLabs/endee) and Python.

## Overview
The project ingests text documents, generates embeddings using `sentence-transformers`, stores them in Endee, and provides a Streamlit interface for semantic search.

## Features
- **Endee Integration**: Custom Python client for Endee REST API.
- **Data Ingestion**: Automated embedding and indexing of text files.
- **Interactive UI**: Streamlit-based search interface.
- **Dockerized Database**: Easy setup using Docker Compose.

## Prerequisites
- Docker & Docker Compose
- Python 3.9+

## Setup & Execution

1. **Start Endee Database**
   ```bash
   docker-compose up -d
   ```
   Wait for the container to start. You can check status with `docker ps`.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ingest Data**
   Run the ingestion script to create the index and upload sample documents:
   ```bash
   python src/ingest.py
   ```

4. **Run the App**
   Start the Streamlit interface:
   ```bash
   streamlit run src/app.py
   ```

5. **Usage**
   - Open your browser at the URL provided by Streamlit (usually http://localhost:8501).
   - Enter a query (e.g., "What is Endee?").
   - View retrieved documents and similarity scores.

## Project Structure
- `src/client.py`: Python wrapper for Endee REST API.
- `src/ingest.py`: Script to process and upload data.
- `src/app.py`: Streamlit application.
- `docker-compose.yml`: Endee server configuration.
- `doc_mapping.json`: Local mapping of document IDs to text content (since we only store vectors in this demo).

## Technical Details
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions).
- **Index Config**: generic `l2` distance, `M=16`, `ef_con=200`.
- **API Endpoints Used**:
  - `POST /api/v1/index/create`
  - `POST /api/v1/index/<name>/vector/insert`
  - `POST /api/v1/index/<name>/search`
