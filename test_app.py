import requests

url = "http://127.0.0.1:5000/predict"
data = {"review": "The product was amazing, I loved it!"}
response = requests.post(url, json=data)
print(response.json())