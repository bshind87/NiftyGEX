"""Word-report helpers shared by the progress and final reports: document setup, tables, figures, bullets, diagrams."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyBboxPatch

from niftygex_eval.figures import SERIES, SURFACE, TEXT, TEXT_2, save

HEADER_FILL = "E8EEF6"
MUTED = RGBColor(0x52, 0x51, 0x4E)
AUTHORS = "Bhalchandra Shinde and Asad Mulani"


def new_document(title: str, subtitle: str):
    """US Letter, 1-inch margins, Calibri 11, blue headings, and the title block."""
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(1))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    for level, size in ((1, 15), (2, 12.5)):
        style = doc.styles[f"Heading {level}"]
        style.font.name, style.font.size, style.font.color.rgb = "Calibri", Pt(size), RGBColor(0x1C, 0x5C, 0xAB)
    run = doc.add_paragraph().add_run(title)
    run.bold, run.font.size = True, Pt(22)
    run = doc.add_paragraph().add_run(subtitle)
    run.font.size, run.font.color.rgb = Pt(13), MUTED
    run = doc.add_paragraph().add_run(f"{AUTHORS} · CS 7150 Deep Learning, Northeastern University · {date.today():%d %B %Y}")
    run.font.size, run.font.color.rgb = Pt(10), MUTED
    return doc


def footer(doc, text: str) -> None:
    run = doc.sections[0].footer.paragraphs[0].add_run(text)
    run.font.size, run.font.color.rgb = Pt(8), MUTED


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def shade(cell, fill: str) -> None:
    pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:color"), "auto")
    element.set(qn("w:fill"), fill)
    pr.append(element)


def table(doc, header: list[str], rows: list[list], widths: list[float], size: float = 9) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.size = True, Pt(size)
        shade(cell, HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(size)
    for row in t.rows:
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
    doc.add_paragraph()


def bullets(doc, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def figure(doc, path: Path, caption: str, width: float = 6.3) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    run = p.add_run(caption)
    run.italic, run.font.size, run.font.color.rgb = True, Pt(9), MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def pipeline_diagram(stages: list[tuple[str, str]], directory: Path) -> Path:
    """Left-to-right stage flow with each stage's main output count."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, len(stages) * 2.0)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (name, detail) in enumerate(stages):
        x = i * 2.0 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.7, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.2,
                                    edgecolor=SERIES[0], facecolor=SURFACE))
        ax.text(x + 0.85, 1.85, name, ha="center", va="center", fontsize=9.5, color=TEXT, weight="bold", wrap=True)
        ax.text(x + 0.85, 1.3, detail, ha="center", va="center", fontsize=7.8, color=TEXT_2, linespacing=1.3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.95, 1.55), xytext=(x + 1.75, 1.55), arrowprops=dict(arrowstyle="-|>", color=TEXT_2, lw=1.2))
    ax.set_title("Data pipeline: Parquet in, Parquet out; each stage refuses stale upstream configuration", fontsize=10.5, loc="left")
    return save(fig, directory, "report_pipeline")
