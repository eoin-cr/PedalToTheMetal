import requests

with open("0-chill.csv", "tr", encoding="utf8") as F:
    text = F.read()
r = requests.post("http://10.33.134.164:5000/post", data={'csv_as_str': text})
print(r.text)