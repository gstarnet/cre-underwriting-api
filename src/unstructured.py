from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable, List, Optional

import pandas as pd

UNSTRUCTURED_TEXT_COL = "unstructured_text"

_TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log"}
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def parse_document_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]

        parts = raw.replace("\n", ",").replace(";", ",").split(",")
        return [p.strip() for p in parts if p.strip()]

    return [str(value).strip()]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise ValueError(f"PDF parsing requires pypdf ({path}): {e}")

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception as e:
        raise ValueError(f"DOCX parsing requires python-docx ({path}): {e}")

    doc = Document(str(path))
    return "\n".join(par.text for par in doc.paragraphs)


def _read_image_ocr(path: Path) -> str:
    try:
        from PIL import Image
    except Exception as e:
        raise ValueError(f"Image OCR requires Pillow ({path}): {e}")

    try:
        import pytesseract
    except Exception as e:
        raise ValueError(f"Image OCR requires pytesseract ({path}): {e}")

    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        raise ValueError(f"Image OCR failed for {path}: {e}")


def extract_text_from_file(path_like: str, *, base_path: Optional[Path] = None) -> str:
    path = Path(path_like)
    if not path.is_absolute() and base_path is not None:
        path = (base_path / path).resolve()

    if not path.exists():
        raise ValueError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext in _TEXT_EXTS:
        text = _read_text_file(path)
    elif ext in _PDF_EXTS:
        text = _read_pdf(path)
    elif ext in _DOCX_EXTS:
        text = _read_docx(path)
    elif ext in _IMAGE_EXTS:
        text = _read_image_ocr(path)
    else:
        raise ValueError(f"Unsupported document type: {path.suffix or '(no extension)'} ({path})")

    return normalize_text(text)


def combine_unstructured_text(
    *,
    inline_text: Optional[str] = None,
    document_paths: Optional[Iterable[str]] = None,
    base_path: Optional[Path] = None,
    strict: bool = False,
    max_chars: int = 50_000,
) -> str:
    chunks: List[str] = []

    inline = normalize_text(inline_text)
    if inline:
        chunks.append(inline)

    for p in document_paths or []:
        try:
            t = extract_text_from_file(str(p), base_path=base_path)
            if t:
                chunks.append(t)
        except Exception:
            if strict:
                raise

    text = " ".join(chunks)
    return text[:max_chars]


def ensure_unstructured_text_column(
    df: pd.DataFrame,
    *,
    inline_text_col: str = "deal_notes",
    document_paths_col: str = "document_paths",
    output_col: str = UNSTRUCTURED_TEXT_COL,
    base_path: Optional[Path] = None,
    strict: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if output_col in out.columns:
        out[output_col] = out[output_col].map(normalize_text)
        return out

    inline = out[inline_text_col] if inline_text_col in out.columns else pd.Series([""] * len(out))
    doc_paths = (
        out[document_paths_col].map(parse_document_paths)
        if document_paths_col in out.columns
        else pd.Series([[]] * len(out))
    )

    out[output_col] = [
        combine_unstructured_text(
            inline_text=inline.iloc[i],
            document_paths=doc_paths.iloc[i],
            base_path=base_path,
            strict=strict,
        )
        for i in range(len(out))
    ]
    return out
