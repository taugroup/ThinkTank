from datetime import datetime
from typing import Dict, List
import io
import re

from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import markdown2

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

def _docx_bytes(project: str,
                desc: str,
                scientists: List[Dict[str, str]],
                meeting: Dict[str, any]) -> bytes:
    doc = Document()
    doc.add_heading(project, level=0)
    doc.add_paragraph(desc)

    doc.add_heading("Scientists", level=1)
    table = doc.add_table(rows=1, cols=4)
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "Title", "Expertise", "Goal", "Role"
    for s in scientists:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text, row[3].text = (
            s["title"], s["expertise"], s["goal"], s["role"]
        )

    doc.add_heading(f"Meeting: {meeting['topic']}", level=1)
    doc.add_paragraph(f"Rounds: {meeting['rounds']}")
    doc.add_heading("Transcript", level=2)
    for m in meeting["transcript"]:
        doc.add_paragraph(f"{m['name']}: {m['content']}")

    doc.add_heading("Summary", level=2)
    doc.add_paragraph(meeting["summary"])

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def _rtf_bytes(project: str,
               desc: str,
               scientists: List[Dict[str, str]],
               meeting: Dict[str, any]) -> bytes:
    """Return a very basic RTF (manual string build, no external lib)."""
    def esc(txt: str) -> str:
        return txt.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

    lines = [
        r"{\rtf1\ansi\deff0",
        r"\fs32\b " + esc(project) + r"\b0\par",
        r"\fs24 " + esc(desc) + r"\par\par",
        r"\b Scientists\b0\par",
    ]
    for s in scientists:
        lines.append(esc(f"{s['title']} | {s['expertise']} | {s['goal']} | {s['role']}") + r"\par")
    lines.extend([
        r"\par\b Meeting: " + esc(meeting["topic"]) + r"\b0\par",
        esc(f"Rounds: {meeting['rounds']}") + r"\par\par",
        r"\b Transcript\b0\par",
    ])
    for m in meeting["transcript"]:
        lines.append(esc(f"{m['name']}: {m['content']}") + r"\par")
    lines.extend([r"\par\b Summary\b0\par", esc(meeting["summary"]), r"\par}", ""])
    return "\n".join(lines).encode("utf-8")

def _pdf_bytes(project: str,
               desc: str,
               scientists: List[Dict[str, str]],
               meeting: Dict[str, any]) -> bytes:
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    elements = []

    # Project title and description
    elements.append(Paragraph(f"<b>{project}</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(desc, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Scientists Table
    elements.append(Paragraph("Scientists", styles['Heading1']))
    data = [["Title", "Expertise", "Goal", "Role"]]
    for s in scientists:
        row = [
            Paragraph(s["title"], styles['Normal']),
            Paragraph(s["expertise"], styles['Normal']),
            Paragraph(s["goal"], styles['Normal']),
            Paragraph(s["role"], styles['Normal']),
        ]
        data.append(row)

    # Adjust column widths to prevent overflow
    col_widths = [1.5*inch, 2.5*inch, 2.5*inch, 1.5*inch]
    table = Table(data, colWidths=col_widths, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Meeting Topic
    elements.append(Paragraph(f"Meeting: {meeting['topic']}", styles['Heading1']))
    elements.append(Paragraph(f"Rounds: {meeting['rounds']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Transcript with markdown parsing
    elements.append(Paragraph("Transcript", styles['Heading2']))
    for m in meeting["transcript"]:
        name = f"<b>{m['name']}:</b> "
        content_html = markdown2.markdown(m["content"])
        content_clean = content_html.replace("<p>", "").replace("</p>", "").strip()
        elements.append(Paragraph(name + content_clean, styles['Normal']))
        elements.append(Spacer(1, 6))
    elements.append(Spacer(1, 12))

    # Summary
    elements.append(Paragraph("Summary", styles['Heading2']))
    elements.append(Paragraph(meeting["summary"], styles['Normal']))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def export_meeting(project_name: str,
                   project_desc: str,
                   scientists: List[Dict[str, str]],
                   meeting: Dict[str, any]) -> Dict[str, bytes]:
    """
    Returns {'docx': …, 'rtf': …, 'pdf': …}  - each value is file bytes.
    """
    return {
        "docx": _docx_bytes(project_name, project_desc, scientists, meeting),
        "rtf": _rtf_bytes(project_name, project_desc, scientists, meeting),
        "pdf": _pdf_bytes(project_name, project_desc, scientists, meeting),
    }