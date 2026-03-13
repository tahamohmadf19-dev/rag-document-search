import pydantic
from typing import List


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: List[str]
    source_id: str = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: List[str]
    sources: List[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int