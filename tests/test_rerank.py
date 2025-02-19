import requests
import json

data = {
    "request": {
        "query": "必须要买实名包赔吗",
        # "candidates": ["二次实名是什么"],
        "candidates": ["二次实名是什么", "300立马拍", "200秒了", "必须买包赔吗"],
    }
}
# data = json.dumps(data, ensure_ascii=False)
# print(data)
# data = data.encode(encoding="utf-8")
url = "http://localhost:8001/reranker"
# res = requests.post(url=url, data=data, headers={"Content-Type": "application/json"})
res = requests.post(url=url, json=data, headers={"Content-Type": "application/json"})
# print(res)
# print(res.text)
print(res.json())
