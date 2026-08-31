from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    page_number: int
    image_index: int
    image_bytes: bytes
    mime_type: str 

@dataclass
class ParsedDocument:
    doc_id: str
    raw_bytes: bytes
    text: str
    tables_as_markdown: list[str] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)
    images_skipped_no_caption: int = 0  

def caption_image(image_bytes: bytes, mime_type: str = "image/png") -> str | None:
    """
    Captions one image via Gemini vision, once, at ingestion time — this
    is the "caption once, embed the caption as text" pattern from the
    architecture discussion, replacing the NotImplementedError stub that
    was here before. Returns None (not a placeholder string) when
    GEMINI_API_KEY isn't configured or the call fails, so callers can
    distinguish "no caption available" from "captioned as empty," and can
    track how many images were actually skipped (see ParsedDocument.
    images_skipped_no_caption) instead of silently losing content.
    """
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                "Describe this image in one or two sentences, focused on any "
                "data, numbers, or structured content it contains (e.g. "
                "'Bar chart showing claim settlement time by product line, "
                "2023-2025' rather than a generic visual description). This "
                "caption will be the ONLY representation of this image "
                "available for search — be specific about what it shows.",
            ],
        )
        return response.text.strip()
    except Exception as e:  
        logger.warning("Image captioning failed (%s) — image skipped, not silently dropped", str(e)[:200])
        return None


def _table_to_markdown(table_data: list[list[str | None]]) -> str:
    if not table_data:
        return ""
    rows = [[cell if cell is not None else "" for cell in row] for row in table_data]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def parse_text_file(path: str | Path) -> ParsedDocument:
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return ParsedDocument(doc_id=path.stem, raw_bytes=raw, text=text)


def parse_pdf_file(path: str | Path) -> ParsedDocument:
    """
    Real layout-aware extraction: text (reading order preserved per page),
    tables (detected structurally, converted to markdown — not garbled
    linear text the way plain pypdf extraction flattens them), and images
    (extracted and captioned once via Gemini vision).
    """
    try:
        import pymupdf
    except ImportError as e:
        raise RuntimeError("pymupdf not installed — run: pip install pymupdf") from e

    path = Path(path)
    raw = path.read_bytes()
    doc = pymupdf.open(str(path))

    all_text_parts: list[str] = []
    tables_as_markdown: list[str] = []
    image_captions: list[str] = []
    images_skipped = 0

    for page_num, page in enumerate(doc):
        all_text_parts.append(page.get_text())

        try:
            tables = page.find_tables()
            for table in tables:
                extracted = table.extract()
                md = _table_to_markdown(extracted)
                if md:
                    tables_as_markdown.append(md)
        except Exception as e:  
            logger.warning("Table detection failed on page %d (%s)", page_num, str(e)[:200])

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                mime_type = f"image/{base_image['ext']}"
            except Exception as e:  
                logger.warning("Image extraction failed on page %d image %d (%s)", page_num, img_index, str(e)[:200])
                images_skipped += 1
                continue

            caption = caption_image(image_bytes, mime_type)
            if caption:
                image_captions.append(f"[Image on page {page_num + 1}]: {caption}")
            else:
                images_skipped += 1

    doc.close()

    return ParsedDocument(
        doc_id=path.stem,
        raw_bytes=raw,
        text="\n".join(all_text_parts),
        tables_as_markdown=tables_as_markdown,
        image_captions=image_captions,
        images_skipped_no_caption=images_skipped,
    )


def parse(path: str | Path) -> ParsedDocument:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf_file(path)
    return parse_text_file(path)
