import requests
from bs4 import BeautifulSoup

url = "https://www.youtube.com/results?search_query=%E5%91%A8%E6%9D%B0%E5%80%AB"
request = requests.get(url)
content = request.content
soup = BeautifulSoup(content, "html.parser")
print(soup)