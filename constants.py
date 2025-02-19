DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_BGE_MODEL = "BAAI/bge-reranker-v2-m3"
# DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL = "/mnt/g/models/bge-reranker-v2-m3"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

# for rerank model
rerank_port = 6420
LOCAL_RERANK_SERVICE_URL = f"localhost:{rerank_port}"
LOCAL_RERANK_MODEL_NAME = "rerank"
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 8

PILOT_PATH = "vector_stores"
