import httpx
from bs4 import BeautifulSoup
import numpy as np

base_url = "https://nav.al/"

def get_quotes(search):
    url = base_url+search
    response = httpx.get(url)
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, "html.parser")
        quotes_html = soup.find_all("p")
        quotes = [q.get_text(strip=True) for q in quotes_html]
        quotes = quotes[2:len(quotes) - 3]
        print(quotes[:3])
        np.save(f'data/chunks/{search}_chunks.npy', quotes)

if __name__ == "__main__":
    get_quotes("rich")

