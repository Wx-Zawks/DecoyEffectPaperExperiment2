from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _collect_math_text(node: ET.Element) -> str:
    return "".join(text.strip() for text in node.itertext() if text and text.strip())


def extract_docx(docx_path: Path) -> dict:
    with zipfile.ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    equations: list[str] = []

    for paragraph in root.findall(".//w:p", NS):
        pieces: list[str] = []
        for text_node in paragraph.findall(".//w:t", NS):
            if text_node.text:
                pieces.append(text_node.text)
        for math_node in paragraph.findall(".//m:oMath", NS):
            math_text = _collect_math_text(math_node)
            if math_text:
                equations.append(math_text)
                pieces.append(f"[MATH:{math_text}]")

        paragraph_text = "".join(pieces).strip()
        if paragraph_text:
            paragraph_text = re.sub(r"\s+", " ", paragraph_text)
            paragraphs.append(paragraph_text)

    return {"paragraphs": paragraphs, "equations": equations}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/extract_docx_text.py <paper.docx> [output.json]")

    docx_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else docx_path.with_suffix(".docx.json")
    data = extract_docx(docx_path)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
