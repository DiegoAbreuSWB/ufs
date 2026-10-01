"""
build.py — Builds the clean and the highlighted versions of the manuscript.

  python build.py            -> ContrastiveXAI_FS_R2_clean.pdf, ContrastiveXAI_FS_R2_highlighted.pdf

The marked build typesets every \\rev{...} passage in a near-black colour; this script
then places real PDF highlight annotations (yellow) over those passages.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV_COLOR = 0x282425  # how MuPDF reports cmyk(0,0,0,0.985)


def latex(n=3):
    for _ in range(n):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=HERE, capture_output=True, text=True, errors="ignore")
        if r.returncode != 0:
            lines = r.stdout.splitlines()
            for i, l in enumerate(lines):
                if l.startswith("!"):
                    print("\n".join(lines[i:i + 8]))
                    break
            sys.exit("LaTeX failed")
    out = r.stdout
    print("  pages:", [l for l in out.splitlines() if "Output written" in l][-1].strip())
    print("  overfull boxes:", out.count("Overfull"), "| undefined refs:", out.count("undefined"))


def highlight(src, dst):
    import fitz

    doc = fitz.open(src)
    n = 0
    for page in doc:
        quads = []
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    if s["color"] == REV_COLOR and s["text"].strip():
                        quads.append(fitz.Rect(s["bbox"]))
        for r in quads:
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=(1, 1, 0))
            a.update()
            n += 1
    doc.save(dst, garbage=3, deflate=True)
    print(f"  highlighted spans: {n}")


def main():
    flag = os.path.join(HERE, "marked.flag")
    if os.path.exists(flag):
        os.remove(flag)
    print("clean build")
    latex()
    shutil.copy(os.path.join(HERE, "main.pdf"), os.path.join(HERE, "ContrastiveXAI_FS_R2_clean.pdf"))
    print("marked build")
    open(flag, "w").close()
    latex()
    highlight(os.path.join(HERE, "main.pdf"), os.path.join(HERE, "ContrastiveXAI_FS_R2_highlighted.pdf"))
    os.remove(flag)


if __name__ == "__main__":
    main()
