from typing import List, Optional

import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: List[str]
    pages: List[int] = []
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: List[str]
    sources: List[str]
    scores: List[float] = []


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int
