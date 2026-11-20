"""Generate tensi's brand assets: mascot, icon, wordmark, lockup (SVG + PNG).

    python brand/make_brand.py

The wordmark is Poppins Bold outlined to paths (no font needed to view it) with a little cube
for the dot of the i.
"""

import subprocess
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"

PLUM = "#2B1E3F"      # outlines, wordmark
CREAM = "#FFF1D6"     # front face
TOP = "#FFF9EE"       # top face (lit)
SIDE = "#F3D3A4"      # side face (shade)
VIOLET = "#7C6CF2"    # tensor grid, accents
BLUSH = "#FF8FA3"     # cheeks
LAVENDER = "#EEEAFF"  # icon background


def lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def grid(a, b, c, d, n=3):
    """Lines dividing quadrilateral a-b-c-d (in order) into n x n cells."""
    out = []
    for i in range(1, n):
        t = i / n
        p, q = lerp(a, b, t), lerp(d, c, t)
        out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}"/>')
        p, q = lerp(a, d, t), lerp(b, c, t)
        out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}"/>')
    return "".join(out)


def mascot(x=0, y=0, s=1.0, mood="happy") -> str:
    """Tensi: a soft little tensor cube that sees. Drawn in a 256x256 box."""
    A, B, C, D = (36, 106), (172, 106), (222, 56), (86, 56)     # top face
    E, F = (222, 184), (172, 234)                                 # side face bottom
    G = (36, 234)                                                 # front bottom-left
    poly = lambda *pts: " ".join(f"{p[0]},{p[1]}" for p in pts)
    eyes = (
        # eyes look slightly up-right, toward whatever you're hovering
        f'<ellipse cx="80" cy="164" rx="16" ry="19" fill="{PLUM}"/>'
        f'<ellipse cx="128" cy="164" rx="16" ry="19" fill="{PLUM}"/>'
        f'<circle cx="86" cy="156" r="6" fill="#fff"/><circle cx="134" cy="156" r="6" fill="#fff"/>'
        f'<circle cx="75" cy="172" r="2.6" fill="#fff" opacity=".8"/><circle cx="123" cy="172" r="2.6" fill="#fff" opacity=".8"/>'
    )
    if mood == "wink":
        eyes = (
            f'<ellipse cx="80" cy="164" rx="16" ry="19" fill="{PLUM}"/>'
            f'<circle cx="86" cy="156" r="6" fill="#fff"/><circle cx="75" cy="172" r="2.6" fill="#fff" opacity=".8"/>'
            f'<path d="M113 166 q15 -13 30 0" fill="none" stroke="{PLUM}" stroke-width="7" stroke-linecap="round"/>'
        )
    return f'''<g transform="translate({x} {y}) scale({s})">
  <g stroke="{PLUM}" stroke-width="9" stroke-linejoin="round" stroke-linecap="round">
    <polygon points="{poly(A, B, C, D)}" fill="{TOP}"/>
    <polygon points="{poly(B, C, E, F)}" fill="{SIDE}"/>
    <polygon points="{poly(A, B, F, G)}" fill="{CREAM}"/>
  </g>
  <g stroke="{VIOLET}" stroke-width="3.2" stroke-linecap="round" opacity=".75">
    {grid(A, B, C, D)}
    {grid(B, C, E, F)}
  </g>
  <g stroke="{PLUM}" stroke-width="9" stroke-linejoin="round" fill="none">
    <polygon points="{poly(A, B, C, D)}"/><polygon points="{poly(B, C, E, F)}"/><polygon points="{poly(A, B, F, G)}"/>
  </g>
  {eyes}
  <ellipse cx="58" cy="196" rx="12" ry="7" fill="{BLUSH}" opacity=".75"/>
  <ellipse cx="150" cy="196" rx="12" ry="7" fill="{BLUSH}" opacity=".75"/>
  <path d="M94 193 q10 11 20 0" fill="none" stroke="{PLUM}" stroke-width="6" stroke-linecap="round"/>
  <path d="M236 20 l4 10 l10 4 l-10 4 l-4 10 l-4 -10 l-10 -4 l10 -4 z" fill="{VIOLET}"/>
</g>'''


def text_path(text: str, size: float, x: float, y: float) -> tuple[str, float]:
    """Outline `text` in Poppins Bold; returns (svg path d, advance width)."""
    font = TTFont(FONT)
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    d, pen_x = [], 0.0
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + pen_x, y)))
        d.append(pen.getCommands())
        pen_x += gs[name].width * scale
    return " ".join(d), pen_x


def wordmark(x: float, baseline: float, size: float) -> tuple[str, float]:
    """`tensi` with a tiny violet cube as the dot of the i."""
    body, w1 = text_path("tens", size, x, baseline)
    dotless, w2 = text_path("ı", size, x + w1, baseline)
    # dot of the i: a small isometric cube centred over the dotless i
    cx = x + w1 + w2 / 2
    cy = baseline - size * 0.735
    r = size * 0.105
    top = f"{cx},{cy - r} {cx + r},{cy - r / 2} {cx},{cy} {cx - r},{cy - r / 2}"
    left = f"{cx - r},{cy - r / 2} {cx},{cy} {cx},{cy + r} {cx - r},{cy + r / 2}"
    right = f"{cx + r},{cy - r / 2} {cx},{cy} {cx},{cy + r} {cx + r},{cy + r / 2}"
    cube = (f'<g stroke="{PLUM}" stroke-width="{size * 0.018:.2f}" stroke-linejoin="round">'
            f'<polygon points="{top}" fill="#B3A8FF"/><polygon points="{left}" fill="{VIOLET}"/>'
            f'<polygon points="{right}" fill="#5B4BD6"/></g>')
    return f'<path d="{body} {dotless}" fill="{PLUM}"/>{cube}', w1 + w2


def svg(w, h, inner, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{rect}{inner}</svg>\n'


def write(name: str, content: str, pngs: dict[str, int] | None = None):
    p = HERE / name
    p.write_text(content)
    for png, width in (pngs or {}).items():
        subprocess.run(["rsvg-convert", "-w", str(width), str(p), "-o", str(HERE / png)], check=True)


if __name__ == "__main__":
    write("tensi-mascot.svg", svg(256, 256, mascot()), {"tensi-mascot.png": 512})
    write("tensi-wink.svg", svg(256, 256, mascot(mood="wink")))
    icon = (f'<rect x="8" y="8" width="240" height="240" rx="56" fill="{LAVENDER}"/>'
            + mascot(28, 34, 0.76))
    write("tensi-icon.svg", svg(256, 256, icon), {"tensi-icon.png": 128, "tensi-icon-512.png": 512})
    wm, ww = wordmark(0, 150, 160)
    write("tensi-wordmark.svg", svg(int(ww) + 10, 190, wm))
    lock, lw = wordmark(260, 168, 150)
    write("tensi-logo.svg", svg(int(260 + lw) + 20, 256, mascot(0, 0, 1.0) + lock),
          {"tensi-logo.png": 1200})
    print("written to", HERE)
