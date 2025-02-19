from dataclasses import dataclass
from constants import DEFAULT_RERANKER_MODEL


@dataclass
class Configuration:
    model_name_or_path = DEFAULT_RERANKER_MODEL
    use_fp16: bool = False
    device: str = "GPU"
    max_length: int = 512
    overlap_tokens: int = 20
    batch_size: int = 8
    k: int = 10
