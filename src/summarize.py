"""Summarize extracted PDF text, tables, and images for retrieval indexing."""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_TEXT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.\n"
    "이 요약본은 임베딩되어 원본 텍스트나 표를 검색하는 데 사용됩니다.\n"
    "주어진 내용을 검색에 최적화된 간결한 한국어 요약으로 작성하세요.\n"
    "요약할 내용: {element}"
)

_IMAGE_SUMMARY_PROMPT = (
    "당신은 검색을 위해 이미지를 요약하는 도우미입니다.\n"
    "이 요약은 임베딩되어 원본 이미지를 검색하는 데 사용됩니다.\n"
    "축 이름, 수치 범위, 추세 등 검색에 유용한 구체적인 정보를 포함해 "
    "간결한 한국어 요약을 작성하세요."
)


def summarize_documents(
    documents: list[Document],
    *,
    model: str | None = None,
    max_concurrency: int = 5,
) -> list[Document]:
    """Summarize page/table Documents in Korean, keeping source metadata.

    Used for both page text and table Documents: both go through the same
    prompt, batched together for speed (`max_concurrency` caps concurrent
    OpenAI calls to avoid RateLimitError).
    """
    load_dotenv()
    if not documents:
        return []

    llm = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = _TEXT_SUMMARY_PROMPT | llm | StrOutputParser()

    non_empty = [document for document in documents if document.page_content.strip()]
    summary_by_id = {}
    if non_empty:
        results = chain.batch(
            [{"element": document.page_content} for document in non_empty],
            {"max_concurrency": max_concurrency},
        )
        summary_by_id = {id(document): result for document, result in zip(non_empty, results)}

    return [
        Document(
            page_content=summary_by_id.get(id(document), ""),
            metadata={**document.metadata, "content_type": "summary"},
        )
        for document in documents
    ]


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def summarize_images(
    image_paths: list[Path],
    *,
    model: str | None = None,
) -> tuple[list[Document], list[Document]]:
    """Return (originals, summaries) Documents, one pair per image.

    `originals[i].page_content` is the base64-encoded image bytes, kept so a
    final answer chain can feed the real image back to a vision model.
    `summaries[i].page_content` is a Korean description used for embedding
    and search. The two lists share the same order and length.
    """
    load_dotenv()
    if not image_paths:
        return [], []

    chat = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=1024)

    originals: list[Document] = []
    summaries: list[Document] = []
    for image_path in image_paths:
        image_b64 = _encode_image(image_path)
        response = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": _IMAGE_SUMMARY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ]
                )
            ]
        )
        metadata = {"source": str(image_path)}
        originals.append(
            Document(page_content=image_b64, metadata={**metadata, "content_type": "image"})
        )
        summaries.append(
            Document(
                page_content=response.content,
                metadata={**metadata, "content_type": "image_summary"},
            )
        )
    return originals, summaries
