from time import sleep
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_url = "https://nav.al/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_quotes(search):
    url = base_url+search
    response = httpx.get(url)
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, "html.parser")
        quotes_html = soup.find_all("p")
        quotes = [q.get_text(strip=True) for q in quotes_html]
        quotes = quotes[2:len(quotes) - 3]
        return quotes

def split_into_four(lst):
    n = len(lst)
    size = n // 4

    chunks = [lst[i:i + size] for i in range(0, n, size)]
    return chunks


def scrape(topics):
    sleep_time = 1
    documents = []
    for topic in topics:
        quotes = get_quotes(topic)
        chunks = split_into_four(quotes)
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content="".join(chunk),
                metadata = {"source": f"{topic}_{idx}"}
            ))
        sleep(sleep_time * 2)
        sleep_time += 1 
        
    return documents


def store_chunks(topics):
    documents = scrape(topics)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    chunks_filepath = f"{BASE_DIR}/data/chunks.npy"
    np.save(chunks_filepath, [chunk.page_source for chunk in chunks])

