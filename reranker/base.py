from typing import List
from abc import ABC


class BaseReranker(ABC):
    """
    Base class for reranker
    """

    def rerank(
        self,
        query: str,
        candidates: List[str],
        k: int,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError
