"""Render docs/REPORT.md to docs/REPORT.pdf (Markdown -> HTML + MathJax -> PDF via Playwright).

Usage: python scripts/build_report_pdf.py      (needs: pip install markdown playwright)
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
SOURCE, TARGET = DOCS / "REPORT.md", DOCS / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1b2433; }
h1 { color: #0c2547; font-size: 21pt; line-height: 1.2; margin-top: 0; }
h2 { color: #0c2547; font-size: 15pt; border-bottom: 1px solid #d6dde8; padding-bottom: 3px; margin-top: 26px;
     page-break-after: avoid; }
h3 { color: #0c6a9d; font-size: 12pt; page-break-after: avoid; }
a { color: #0c6a9d; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d6dde8; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef3f9; }
img { max-width: 100%; display: block; margin: 10px auto; page-break-inside: avoid; }
p img { display: inline-block; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f8; padding: 0 3px; border-radius: 3px; }
pre { background: #f2f4f8; padding: 10px; border-radius: 6px; font-size: 8.8pt; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #cc163f; margin: 8px 0; padding: 2px 12px; color: #52607a; }
hr { border: 0; border-top: 1px solid #d6dde8; margin: 18px 0; }
"""

MATHJAX = """
<script>window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
  startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => { document.body.dataset.math = 'ok'; }) } };</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
"""


def protect_math(text: str) -> tuple[str, list[str]]:
    """Keep $...$ / $$...$$ away from the Markdown parser (underscores, backslashes)."""
    stash: list[str] = []

    def keep(m: re.Match[str]) -> str:
        stash.append(m.group(0))
        return f"@@MATH{len(stash) - 1}@@"
    return re.sub(r"\$\$.+?\$\$|\$[^$\n]+\$", keep, text, flags=re.S), stash


def main() -> None:
    text, stash = protect_math(SOURCE.read_text(encoding="utf-8"))
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    body = re.sub(r"@@MATH(\d+)@@", lambda m: stash[int(m.group(1))], body)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><base href='{DOCS.as_uri()}/'>"
            f"<style>{CSS}</style>{MATHJAX}</head><body>{body}</body></html>")
    html_path = DOCS / "_report_render.html"
    html_path.write_text(page, encoding="utf-8")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge")
            pg = browser.new_page()
            pg.goto(html_path.as_uri(), wait_until="networkidle")
            pg.wait_for_selector("body[data-math='ok']", timeout=30_000)
            pg.pdf(path=str(TARGET), format="A4", print_background=True,
                   display_header_footer=True, header_template="<span></span>",
                   footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>"
                                   "Sightector v2 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                   margin={"top": "16mm", "bottom": "18mm", "left": "15mm", "right": "15mm"})
            browser.close()
    finally:
        html_path.unlink(missing_ok=True)
    print("saved", TARGET)


if __name__ == "__main__":
    main()
