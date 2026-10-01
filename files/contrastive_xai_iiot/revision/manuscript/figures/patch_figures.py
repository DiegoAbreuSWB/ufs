"""
patch_figures.py — Text corrections on the two pipeline diagrams (the original vector
sources are not available). Only labels that contradicted the revised text are replaced:

  Fig. 1  "Key Findings" bullets, "High F1-macro", "Interpretable Security Insights" labels
  Fig. 2  objective formula (fixed k*), candidate-k note, noise bullet, cluster-profile labels

Originals are kept as *_orig.png.
"""

import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def bg_color(im, box):
    x0, y0, x1, y1 = box
    a = np.asarray(im.crop(box))
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def background_layer(im, box, strip=5):
    """Row-wise background taken from the pixels just left and right of the box (follows gradients)."""
    x0, y0, x1, y1 = box
    a = np.asarray(im).astype(np.float32)
    left = a[y0:y1, max(0, x0 - strip):x0].reshape(y1 - y0, -1, 3)
    right = a[y0:y1, x1:x1 + strip].reshape(y1 - y0, -1, 3)
    rows = np.median(np.concatenate([left, right], axis=1), axis=1)       # (h, 3)
    med = np.median(rows, axis=0)
    rows = np.where(np.abs(rows - med).sum(1, keepdims=True) > 40, med, rows)  # ignore rows crossing other artwork
    layer = np.repeat(rows[:, None, :], x1 - x0, axis=1)
    return Image.fromarray(layer.clip(0, 255).astype(np.uint8), "RGB")


def put_text(im, box, text, fnt, fill, align="left", blur=0.45, pad=0):
    """Paint `box` with its background colour and draw `text` (soft edges to match the artwork)."""
    x0, y0, x1, y1 = box
    layer = background_layer(im, box)
    d = ImageDraw.Draw(layer)
    lines = text.split("\n")
    heights = [d.textbbox((0, 0), ln, font=fnt)[3] for ln in lines]
    line_h = max(heights) + 3
    total = line_h * len(lines)
    y = (y1 - y0 - total) // 2
    for ln in lines:
        w = d.textbbox((0, 0), ln, font=fnt)[2]
        x = pad if align == "left" else (x1 - x0 - w) // 2
        d.text((x, y), ln, font=fnt, fill=fill)
        y += line_h
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    im.paste(layer, (x0, y0))


def math_image(tex, height_px, color="#1c1c1c"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 1), dpi=300)
    fig.text(0.5, 0.5, tex, fontsize=20, ha="center", va="center", color=color, math_fontfamily="stix")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    m = Image.open(buf).convert("RGBA")
    w = int(m.width * height_px / m.height)
    return m.resize((w, height_px), Image.LANCZOS)


def put_math(im, box, tex, height_px):
    x0, y0, x1, y1 = box
    layer = background_layer(im, box).convert("RGBA")
    m = math_image(tex, height_px)
    layer.alpha_composite(m, ((x1 - x0 - m.width) // 2, (y1 - y0 - m.height) // 2))
    im.paste(layer.convert("RGB").filter(ImageFilter.GaussianBlur(0.35)), (x0, y0))


def fig1():
    src = os.path.join(HERE, "fig1_overview_orig.png")
    if not os.path.exists(src):
        os.replace(os.path.join(HERE, "fig1_overview.png"), src)
    im = Image.open(src).convert("RGB")
    dark = (28, 32, 44)
    f = font("segoeui.ttf", 20)
    # Key Findings -> result-independent properties of the pipeline
    put_text(im, (120, 580, 330, 619), "Key Properties", font("seguisb.ttf", 24), (29, 78, 137), align="center", blur=0.4)
    put_text(im, (88, 644, 410, 676), "Label-free feature refinement", f, dark, pad=5)
    put_text(im, (88, 699, 410, 731), "Fold-internal, leakage-free protocol", f, dark, pad=5)
    put_text(im, (88, 754, 410, 786), "Held-out cluster explanations", f, dark, pad=5)
    # downstream box
    put_text(im, (590, 762, 840, 800), "Macro-F1 on test folds", font("seguisb.ttf", 22), (88, 50, 150), pad=14)
    # insights box
    put_text(im, (1366, 572, 1640, 604), "Behavioral Cluster Profiles", font("seguisb.ttf", 20), (255, 255, 255), align="center", blur=0.4)
    f2 = font("seguisb.ttf", 19)
    for y, label in ((622, "Dominant features"), (674, "Held-out fidelity"), (726, "Clarity index"), (779, "Analyst cues")):
        put_text(im, (1436, y, 1626, y + 34), label, f2, dark, pad=8)
    im.save(os.path.join(HERE, "fig1_overview.png"))


def fig2():
    src = os.path.join(HERE, "fig2_pipeline_orig.png")
    if not os.path.exists(src):
        os.replace(os.path.join(HERE, "fig2_pipeline.png"), src)
    im = Image.open(src).convert("RGB")
    dark = (28, 32, 44)
    # clustering evaluation note: fixed k*
    put_text(im, (912, 172, 1074, 194), "(at the fixed k* of the fold)", font("seguisb.ttf", 12), dark, align="center", blur=0.4)
    put_math(im, (912, 194, 1074, 217), r"$k^{*} \in \{2,\ldots,14\}$", 19)
    # objective
    put_math(im, (648, 400, 1016, 437), r"$J(S)\;=\;\mathrm{Sil}\,\left(H_S,\ \Pi_{k^{*}}(H_S)\right)$", 31)
    # augmentation bullet
    put_text(im, (474, 338, 572, 378), "Add Gaussian\nnoise", font("seguisb.ttf", 12), dark, pad=1, blur=0.4)
    # cluster profile labels
    f2 = font("seguisb.ttf", 15)
    for y, label in ((587, "Dominant features"), (645, "Held-out fidelity"), (700, "Clarity index"), (756, "Analyst cues")):
        put_text(im, (1480, y, 1618, y + 26), label, f2, dark, pad=4)
    im.save(os.path.join(HERE, "fig2_pipeline.png"))


if __name__ == "__main__":
    fig1()
    fig2()
    print("figures patched")
