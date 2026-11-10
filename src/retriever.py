"""Create a multi-vector retriever over text, table, and image summaries."""

import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings


def create_multi_vector_retriever(
    originals: list[Document],
    summaries: list[Document],
    *,
    k: int = 6,
) -> MultiVectorRetriever:
    """Index summaries for search; return the matching original documents.

    `originals` and `summaries` must be the same length, each entry describing
    the same unit of content (a text block, a table, or an image). An image's
    original is its base64-encoded bytes with metadata `content_type="image"`,
    so a final answer chain can feed it back to a vision model.
    """
    if len(originals) != len(summaries):
        raise ValueError("originals and summaries must be the same length.")
    if not originals:
        raise ValueError("At least one document is required to build the retriever.")

    load_dotenv()
    vectorstore = Chroma(
        collection_name=f"multimodal-rag-{uuid4().hex}",
        embedding_function=OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        ),
    )
    docstore = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": k},
    )

    doc_ids = [uuid4().hex for _ in originals]
    summary_docs = [
        Document(page_content=summary.page_content, metadata={**summary.metadata, id_key: doc_id})
        for doc_id, summary in zip(doc_ids, summaries)
    ]
    vectorstore.add_documents(summary_docs)
    docstore.mset(list(zip(doc_ids, originals)))
    return retriever
