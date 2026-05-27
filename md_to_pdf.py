"""Render report.md (or any Markdown file) into a searchable PDF.

Uses the `markdown-pdf` package (HTML → PDF via PyMuPDF), which embeds
selectable text — addresses the HW1 feedback that the PDF must be
searchable.

Usage::

    python md_to_pdf.py --input report.md --output 314560017_HW3.pdf
"""

from __future__ import annotations

import argparse
import os
import re

from markdown_pdf import MarkdownPdf, Section


def _strip_yaml_front_matter(text: str) -> tuple[str, dict]:
    """Tiny YAML parser that supports both `key: value` and list-style
    keys (`key:` followed by `  - item` lines). Values returned for
    list keys are joined with `; `."""
    meta: dict = {}
    if not text.startswith("---"):
        return text, meta
    end = text.find("\n---", 3)
    if end == -1:
        return text, meta

    block = text[3:end]
    body = text[end + 4:].lstrip("\n")

    current_key: str | None = None
    current_list: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # list item under the previous key
        if line.lstrip().startswith("- ") and current_key is not None:
            current_list.append(line.lstrip()[2:].strip().strip('"'))
            continue
        # close any open list
        if current_key is not None and current_list:
            meta[current_key] = "; ".join(current_list)
            current_key, current_list = None, []
        m = re.match(r"^([A-Za-z][\w-]*)\s*:\s*(.*)$", line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip().strip('"')
            if val == "":
                current_key, current_list = key, []
            else:
                meta[key] = val
    # flush any trailing list
    if current_key is not None and current_list:
        meta[current_key] = "; ".join(current_list)

    return body, meta


def _strip_pandoc_image_attrs(text: str) -> str:
    """Pandoc-style image width attrs `{ width=80% }` are not understood
    by markdown-pdf — drop them so the image renders at natural size."""
    return re.sub(r"\{\s*width=[^}]*\}", "", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="report.md")
    parser.add_argument("--output", default="314560017_HW3.pdf")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        md = f.read()
    md, meta = _strip_yaml_front_matter(md)
    md = _strip_pandoc_image_attrs(md)

    css = """
    body { font-family: 'Segoe UI', 'Helvetica', sans-serif;
           font-size: 11pt; line-height: 1.45; color: #222; }
    h1 { font-size: 20pt; margin-top: 18pt; border-bottom: 1px solid #ccc;
         padding-bottom: 4pt; }
    h2 { font-size: 15pt; margin-top: 14pt; color: #1f3a93; }
    h3 { font-size: 12.5pt; margin-top: 10pt; color: #2c3e50; }
    code { background: #f4f4f4; padding: 0 3px; border-radius: 3px;
           font-size: 90%; }
    pre  { background: #f4f4f4; padding: 8px; border-radius: 4px;
           font-size: 9.5pt; overflow-x: auto; }
    table { border-collapse: collapse; margin: 8pt 0; }
    th, td { border: 1px solid #bbb; padding: 4pt 8pt; font-size: 10pt; }
    th { background: #f0f0f0; }
    img { max-width: 100%; height: auto; display: block; margin: 6pt auto; }
    blockquote { border-left: 3px solid #888; color: #555;
                 padding-left: 10pt; margin-left: 0; font-style: italic; }
    a { color: #1f3a93; text-decoration: none; }
    """

    title = meta.get("title", "Report")
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", "")

    # markdown-pdf builds its TOC from h1..h6 and requires the first
    # heading to be h1. Prepend a synthetic title block.
    header_lines = [f"# {title}"]
    if subtitle:
        header_lines.append(f"_{subtitle}_")
    if author:
        header_lines.append("")
        header_lines.append(author)
    md = "\n".join(header_lines) + "\n\n" + md

    pdf = MarkdownPdf()
    pdf.meta["title"] = title
    if author:
        pdf.meta["author"] = author
    pdf.add_section(Section(md, root=os.path.dirname(os.path.abspath(args.input)) or "."), user_css=css)
    pdf.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
