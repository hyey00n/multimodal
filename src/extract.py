"""Split a PDF into text blocks, tables, and images using Unstructured's layout parser."""

from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_pdf(
    pdf_path: str | Path,
    figure_dir: str | Path | None = None,
) -> tuple[list[Document], list[Document], list[Path]]:
    """Return (texts, tables, image_paths) extracted from a PDF.

    Runs Unstructured's `hi_res` layout model so tables are recognized as
    distinct elements (not flattened into text) and embedded images are
    written out as files. Requires Poppler and Tesseract (with the `kor`
    language pack) on PATH.
    """
    source_path = Path(pdf_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source_path}")

    output_dir = Path(figure_dir) if figure_dir else PROJECT_ROOT / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(source_path),
        strategy="hi_res",
        infer_table_structure=True,
        languages=["kor", "eng"],
        extract_image_block_types=["Image"],
        extract_image_block_output_dir=str(output_dir),
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    texts: list[Document] = []
    tables: list[Document] = []
    for element in elements:
        metadata = {
            "source": str(source_path),
            "page": getattr(element.metadata, "page_number", None),
        }
        element_type = type(element).__name__
        if element_type == "Table":
            metadata["content_type"] = "table"
            metadata["html"] = getattr(element.metadata, "text_as_html", None)
            tables.append(Document(page_content=str(element), metadata=metadata))
        elif element_type == "CompositeElement":
            metadata["content_type"] = "text"
            texts.append(Document(page_content=str(element), metadata=metadata))

    image_paths = sorted(
        path for path in output_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    return texts, tables, image_paths
