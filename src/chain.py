"""Generate a final answer from retrieved text, table, and image context."""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


def _split_context(documents: list[Document]) -> dict:
    images, texts = [], []
    for document in documents:
        if document.metadata.get("content_type") == "image":
            images.append(document.page_content)
        else:
            texts.append(document.page_content)
    return {"images": images, "texts": texts}


def _build_prompt(data: dict) -> list[HumanMessage]:
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
        for image in data["context"]["images"]
    ]
    formatted_texts = "\n\n".join(data["context"]["texts"])
    content.append(
        {
            "type": "text",
            "text": (
                "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
                "자료에 없는 내용은 모른다고 답하세요.\n\n"
                f"질문: {data['question']}\n\n"
                f"참고 자료:\n{formatted_texts}"
            ),
        }
    )
    return [HumanMessage(content=content)]


def build_answer_chain(retriever, model: str | None = None) -> Runnable:
    """Return a chain that retrieves context and generates a grounded answer.

    The retriever's hits are split into base64 images and text/table
    excerpts, then sent together to a vision model so the answer can cite
    details that only appear in a chart or table, not just the body text.
    """
    load_dotenv()
    llm = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), max_tokens=1024)

    return (
        {
            "context": retriever | RunnableLambda(_split_context),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(_build_prompt)
        | llm
        | StrOutputParser()
    )
