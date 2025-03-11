#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author:     sai.chen
@FileName:   api.py
@Date:       2025/02/17
@Description:
===============================================
Rerankers API

# import sys
# sys.path.append("..")
# # 获取当前脚本的绝对路径
# current_script_path = os.path.abspath(__file__)
# # 将项目根目录添加到sys.path
# root_dir = os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# )
# sys.path.append(root_dir)
# pwd_path = os.path.abspath(os.path.dirname(__file__))
#
===============================================
"""


import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from reranker.rerankers import MyRerankerModel
from log import get_logger
from configs import Configuration

logger = get_logger("reranker")
# import logging

# logging.basicConfig(
#     format="%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__file__)


class Item(BaseModel):
    """传入数据的指定格式"""

    # input: str = Field(..., max_length=512)
    request: dict = {}


# rerank_model
logger.info("loadding rerank_model ...")
config = Configuration
bce_reranker = MyRerankerModel(config)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return """Main Server Start!\n You can get reranker here."""


@app.post("/reranker")
async def hf_rerank(data: Item):
    """根据用户query对候选相似数据重排

    query: 当前用户query, str
    candidates: 候选相似数据, list

    return:
        scores: list
    """
    input_data = data.request
    query = input_data.get("query")
    logger.info(f"query: {query}")
    candidates = input_data.get("candidates")
    logger.info(f"candidates: {candidates}")
    if not query or not candidates:
        return {"error": "Both 'query' and 'candidates' are required."}, 400

    # Rerank
    rank_results = bce_reranker.rerank(query, candidates=candidates)
    # print(f"rank_results: \n{rank_results}")
    return rank_results


if __name__ == "__main__":
    logger.info(f"use cuda ? :{torch.cuda.is_available()}")
    uvicorn.run(app=app, host="0.0.0.0", port=8001, workers=1)
    # uvicorn.run(app="api:app", host='0.0.0.0', port=8001, reload=True, workers=1)
