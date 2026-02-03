import requests
from typing import List, Dict, Optional, Union, Any

class EndeeClient:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = api_key

    def _get_url(self, endpoint: str) -> str:
        return f"{self.base_url}/api/v1{endpoint}"

    def health(self) -> Dict:
        """Check if the server is running."""
        response = requests.get(self._get_url("/health"))
        response.raise_for_status()
        return response.json()

    def create_index(self, index_name: str, dim: int, space_type: str = "l2", m: int = 16, ef_con: int = 200, precision: str = "float32") -> Dict:
        """
        Create a new index.
        space_type: l2, cosine, ip
        precision: float32, float16, int8
        """
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "M": m,
            "ef_con": ef_con,
            "precision": precision
        }
        response = requests.post(self._get_url("/index/create"), json=payload, headers=self.headers)
        if response.status_code == 409:
             print(f"Index {index_name} might already exist or conflict.")
        response.raise_for_status()
        return response.json() if response.content else {}

    def list_indexes(self) -> Dict:
        """List all indexes for the current user."""
        response = requests.get(self._get_url("/index/list"), headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_index(self, index_name: str) -> bool:
        """Delete an index."""
        response = requests.delete(self._get_url(f"/index/{index_name}/delete"), headers=self.headers)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """
        Insert vectors into an index.
        vectors list of dicts:
        [
            {"id": "1", "vector": [0.1, ...]},
            {"id": "2", "vector": [...], "sparse_indices": [...], "sparse_values": [...]}
        ]
        """
        # The API expects a list of objects or a single object.
        # Based on code: CROW_ROUTE(app, "/api/v1/index/<string>/vector/insert")
        url = self._get_url(f"/index/{index_name}/vector/insert")
        response = requests.post(url, json=vectors, headers=self.headers)
        response.raise_for_status()
        return response.status_code == 200

    def search(self, index_name: str, vector: List[float], k: int = 10, include_vectors: bool = False) -> Any:
        """
        Search for nearest neighbors.
        """
        payload = {
            "vector": vector,
            "k": k,
            "include_vectors": include_vectors
        }
        url = self._get_url(f"/index/{index_name}/search")
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        # The response is msgpack by check in main.cpp, but let's see if requests handles it or if I need msgpack python lib.
        # The server code says: resp.add_header("Content-Type", "application/msgpack");
        # Implementation Detail: If python requests doesn't auto-decode msgpack, we might need `msgpack.unpackb`.
        # However, many REST APIs support JSON if asked or fallback. 
        # Looking at server code: 
        # msgpack::pack(sbuf, search_response.value());
        # It ONLY sends msgpack. 
        # So I need to use msgpack to decode.
        import msgpack
        return msgpack.unpackb(response.content, raw=False)

