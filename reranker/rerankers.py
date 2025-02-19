#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author:     sai.chen
@FileName:   rerankers.py
@Date:       2025/02/17
@Description: 
===============================================
rerankers of HF backend and trition backend
===============================================
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseReranker
from tqdm import tqdm
from typing import List, Tuple, Union, Any
import logging

logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


class MyRerankerModel(BaseReranker):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        # model_path = "/mnt/g/models/bge-reranker-v2-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, **kwargs
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path, **kwargs
        )
        # logger.info(f"Loading from `{model_name_or_path}`.")

        self.num_gpus = torch.cuda.device_count()
        self.config.device = "cuda" if self.num_gpus > 0 else "cpu"

        if self.config.device == "cpu":
            self.num_gpus = 0
        elif self.config.device.startswith("cuda:") and self.num_gpus > 0:
            self.num_gpus = 1
        elif self.config.device == "cuda":
            self.num_gpus = self.num_gpus
        else:
            raise ValueError(
                "Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !"
            )

        if self.config.use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.config.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            # batch inference
            self.batch_size = self.config.batch_size * self.num_gpus

        # for advanced preproc of tokenization
        self.max_length = self.config.max_length
        self.overlap_tokens = self.config.overlap_tokens  # 80

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        enable_tqdm: bool = False,
        **kwargs,
    ):

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(
                range(0, len(sentence_pairs), self.batch_size),
                desc="Calculate scores",
                disable=not enable_tqdm,  # disable
            ):
                sentence_pairs_batch = sentence_pairs[
                    sentence_id : sentence_id + self.batch_size
                ]
                inputs = self.tokenizer(
                    sentence_pairs_batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                inputs_on_device = {
                    k: v.to(self.config.device) for k, v in inputs.items()
                }
                scores = (
                    self.model(**inputs_on_device, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
                scores = torch.sigmoid(scores)
                scores_collection.extend(scores.cpu().numpy().tolist())

        return scores_collection

    def rerank(
        self,
        query: str,
        candidates: List[str],
        **kwargs,
    ) -> list[dict[str, Any]]:

        # remove invalid candidates
        candidates = [p[:5000] for p in candidates if isinstance(p, str) and 0 < len(p)]
        if query is None or len(query) == 0 or len(candidates) == 0:
            return []

        sentence_pairs = [[query, candidate] for candidate in candidates]
        scores = self.compute_score(sentence_pairs)
        # sorted_candidates = np.argsort(merge_scores)[::-1]
        sorted_candidates = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        # return sorted_candidates
        top_k = [
            {"text": doc, "score": score}
            for doc, score in sorted_candidates[: self.config.k]
        ]
        return top_k
