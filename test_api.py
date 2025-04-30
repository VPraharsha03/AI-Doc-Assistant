import requests
url = "http://localhost:8000/query"
data = {"query": "What are the physical characteristics of the Classic Yeti?"}
response = requests.post(url, json=data)
print(requests.get("http://localhost:8000/"))
print(requests.get("http://localhost:8000/info"))
print(response.json())