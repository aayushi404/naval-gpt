import os
import numpy as np

def load_files():
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    print(BASE_DIR)
    entries = os.listdir(os.path.join(BASE_DIR, "data", "embeddings"))
    embedding_files = [f for f in entries if os.path.isfile(os.path.join(BASE_DIR,"data/embeddings/", f))]
    print(embedding_files)
    embeddings = []
    chunks = []
    for f in embedding_files:
        data = np.load(os.path.join(BASE_DIR,"data/embeddings/", f))
        embeddings.extend(data['embeddings'])
        chunks.extend(data['chunks'])
    return embeddings, chunks
