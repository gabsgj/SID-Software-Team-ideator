#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    from markdown import markdown
except Exception:
    print('Missing Python dependency: markdown')
    raise

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except Exception:
    print('Missing Python dependency: reportlab')
    raise


def md_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding='utf-8')

    # Very simple Markdown -> text handling: keep headings, code blocks, and paragraphs
    lines = text.splitlines()
    story = []
    styles = getSampleStyleSheet()
    hstyle = ParagraphStyle('Heading', parent=styles['Heading1'], spaceAfter=12)
    code_style = ParagraphStyle('Code', parent=styles['Code'])
    pre_style = ParagraphStyle('Pre', fontName='Courier', fontSize=8, leading=10)
    normal = styles['Normal']

    in_code = False
    code_block = []

    for ln in lines:
        if ln.strip().startswith('```'):
            if not in_code:
                in_code = True
                code_block = []
            else:
                # end code block
                in_code = False
                story.append(Preformatted('\n'.join(code_block), pre_style))
                story.append(Spacer(1, 6))
            continue
        if in_code:
            code_block.append(ln)
            continue

        if ln.startswith('#'):
            level = len(ln) - len(ln.lstrip('#'))
            text = ln.lstrip('#').strip()
            # use heading style
            story.append(Paragraph(text, hstyle))
            story.append(Spacer(1, 6))
            continue

        if ln.strip() == '---' or ln.strip() == '***':
            story.append(Spacer(1, 12))
            continue

        if ln.strip() == '':
            story.append(Spacer(1, 6))
            continue

        # fallback paragraph
        story.append(Paragraph(ln, normal))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=72)
    doc.build(story)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: md_to_pdf.py <input.md> <output.pdf>')
        sys.exit(2)
    md_to_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
