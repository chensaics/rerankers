import json
import requests


# consult_url = "http://121.41.64.235:7500/predict"
consult_url = "http://121.41.64.235:7500/predict"


def post_consult_result(query):
    """{
        "code": 200,
        "data": {
            "label": 1,
            "prob": 0.9996719360351562
        },
        "mesg": ""
    }
    """
    data = {"query": query}
    # data = json.dumps(data, ensure_ascii=False)
    # data = data.encode(encoding="utf-8")
    res = requests.post(
        consult_url, json=data, headers={"Content-Type": "application/json"}
    )
    print(res.json())
    return res.json()["data"]


post_consult_result("我考虑一下吧明天交易")
