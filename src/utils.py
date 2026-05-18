import io
import os
import pypdf
import docx2txt

def extract_text(file_source, file_name=None):
    """
    Extracts text from TXT, PDF, or DOCX files.
    file_source can be a string path to a file, or file-like bytes object.
    file_name is required if file_source is a bytes object, to determine file type.
    """
    if file_name is None:
        if isinstance(file_source, str):
            file_name = os.path.basename(file_source)
        else:
            raise ValueError("file_name must be provided if file_source is not a file path string.")

    ext = file_name.split('.')[-1].lower()

    # If file_source is a path, let's open it in appropriate mode
    is_path = isinstance(file_source, str)

    try:
        if ext == 'txt':
            if is_path:
                with open(file_source, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                return file_source.decode('utf-8', errors='ignore')

        elif ext == 'pdf':
            text = ""
            if is_path:
                reader = pypdf.PdfReader(file_source)
            else:
                # Wrap bytes in BytesIO for pypdf
                stream = io.BytesIO(file_source)
                reader = pypdf.PdfReader(stream)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

        elif ext == 'docx':
            if is_path:
                return docx2txt.process(file_source)
            else:
                stream = io.BytesIO(file_source)
                return docx2txt.process(stream)
        else:
            # Fallback for other formats - treat as text
            if is_path:
                with open(file_source, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                return file_source.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error parsing file {file_name}: {e}")
        return ""

def chunk_text(text, chunk_size=800, overlap=150):
    """
    Splits text into clean, overlapping character chunks.
    Ensures word boundaries are respected when possible.
    """
    if not text:
        return []
        
    # Standardize whitespace
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    text_len = len(text)
    
    if text_len <= chunk_size:
        return [text]
        
    while start < text_len:
        end = start + chunk_size
        
        # If we aren't at the end of the text, look for a clean break (space)
        if end < text_len:
            # Look backwards up to 100 characters for a space
            last_space = text.rfind(" ", end - 100, end)
            if last_space != -1:
                end = last_space
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap
        
        # Safety guard to prevent infinite loops if overlap is misconfigured
        if start >= end:
            start = end
            
    return chunks
