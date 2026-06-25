"""OCR block retrieval for scan/image-heavy document questions."""

from __future__ import annotations

import re

from verilume.core.ocr_blocks import OCRBlockStore
from verilume.core.schemas import LocalSource

OCR_QUERY_RE = re.compile(r"\b(?:ocr|scan|scanned|image|text block|readable|extract text)\b", re.IGNORECASE)


def is_ocr_query(question: str) -> bool:
    return bool(OCR_QUERY_RE.search(question or ""))


class OCRRetriever:
    def __init__(self, store: OCRBlockStore) -> None:
        self.store = store

    def retrieve(self, question: str, *, limit: int = 5) -> list[LocalSource]:
        if not is_ocr_query(question):
            return []
        sources: list[LocalSource] = []
        for index, block in enumerate(self.store.search(question, limit=limit), start=1):
            sources.append(
                LocalSource(
                    label=f"S{index}",
                    document=block.document,
                    page=block.page or None,
                    chunk_id=f"ocr:{block.block_id}",
                    text=f"OCR text block: {block.text}",
                    score=float(block.confidence) if block.confidence is not None else 0.55,
                    metadata={
                        "content_type": "ocr_block",
                        "ocr_block_id": block.block_id,
                        "ocr_confidence": block.confidence,
                        "ocr_block_type": block.block_type,
                        "retrieval": "ocr",
                    },
                )
            )
        return sources
