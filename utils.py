from datetime import datetime
from typing import Dict, List
import tempfile
import re, os, copy
import pandas as pd
import pypandoc
pypandoc.download_pandoc()

from docx import Document
from docx.shared import Pt

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_name(name: str, min_len: int = 3) -> str:
    # replace invalid chars with _
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    # collapse consecutive underscores / dots / dashes
    clean = re.sub(r"[_\.-]{2,}", "_", clean)
    # strip leading / trailing non-alphanumerics
    clean = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", clean)
    # fallback if too short
    if len(clean) < min_len:
        clean = (clean + "___")[:min_len]
    return clean[:512]

def indent(text: str, pad: int = 2) -> str:
    prefix = " " * pad
    return "\n".join(prefix + ln for ln in text.splitlines())

def _docx_bytes(project_name: str,
                project_desc: str,
                scientists: List[Dict[str, str]],
                md_text: List[str],
                table_font_size: int = 10) -> bytes:
    md_list = copy.deepcopy(md_text)        # never mutate caller’s list
    table_md = pd.DataFrame(scientists).to_markdown(index=False, tablefmt="pipe")
    md_list.insert(0, table_md)             # put table above any meeting notes

    body_md   = "\n\n".join(md_list)
    header_md = (
        f"# {project_name}\n\n"
        f"---\n\n"
        f"## {project_desc}\n\n"
        f"**Exported on:** {now()}\n"
    )
    full_markdown = f"{header_md}\n\n{body_md}"
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    pypandoc.convert_text(full_markdown, to="docx", format="md", outputfile=tmp_path)

    doc = Document(tmp_path)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(table_font_size)

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as out:
        out_path = out.name
    doc.save(out_path)

    with open(out_path, "rb") as f:
        docx_bytes = f.read()

    os.remove(tmp_path)
    os.remove(out_path)

    return docx_bytes

def export_meeting(project_name: str,
                   project_desc: str,
                   scientists: List[Dict[str, str]],
                   meeting: Dict[str, any],
                   md_text: List[str]) -> Dict[str, bytes]:
    """
    Returns {'docx': …, 'rtf': …, 'pdf': …}  - each value is file bytes.
    """
    return {
        "docx": _docx_bytes(project_name, project_desc, scientists, md_text)
    }