from __future__ import annotations

import pandas as pd
import pytest

from src.unstructured import (
    combine_unstructured_text,
    ensure_unstructured_text_column,
    extract_text_from_file,
    parse_document_paths,
)


def test_parse_document_paths_handles_list_csv_and_json():
    assert parse_document_paths(["a.txt", " b.txt "]) == ["a.txt", "b.txt"]
    assert parse_document_paths("a.txt,b.txt;c.txt") == ["a.txt", "b.txt", "c.txt"]
    assert parse_document_paths('["a.txt", "b.txt"]') == ["a.txt", "b.txt"]


def test_extract_text_from_txt_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("line one\nline   two", encoding="utf-8")

    out = extract_text_from_file(str(p))
    assert out == "line one line two"


def test_extract_text_unsupported_extension_raises(tmp_path):
    p = tmp_path / "file.xyz"
    p.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_text_from_file(str(p))


def test_combine_unstructured_text_inline_and_docs(tmp_path):
    p = tmp_path / "lease.txt"
    p.write_text("anchor tenant 7 years", encoding="utf-8")

    out = combine_unstructured_text(
        inline_text="recent capex completed",
        document_paths=[str(p)],
        strict=True,
    )
    assert "recent capex completed" in out
    assert "anchor tenant 7 years" in out


def test_ensure_unstructured_text_column_from_notes_and_paths(tmp_path):
    p = tmp_path / "deal.txt"
    p.write_text("high rollover risk", encoding="utf-8")

    df = pd.DataFrame(
        [
            {"deal_notes": "stabilized asset", "document_paths": [str(p)]},
            {"deal_notes": None, "document_paths": None},
        ]
    )

    out = ensure_unstructured_text_column(df, strict=True)
    assert "unstructured_text" in out.columns
    assert "stabilized asset" in out.loc[0, "unstructured_text"]
    assert "high rollover risk" in out.loc[0, "unstructured_text"]
    assert out.loc[1, "unstructured_text"] == ""
