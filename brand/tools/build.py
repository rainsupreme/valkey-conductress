#!/usr/bin/env python3
"""Build every Conductress brand asset from one source of truth.

Run from anywhere:  python3 brand/tools/build.py

Outputs (all checked in, so consumers never need to run this):
  brand/logo/conductress-hero.svg          900x340 opaque scene: sky, grid, streaks, trail, C, wordmark, subtitle
  brand/logo/conductress-lockup.svg        900x340 transparent: streaks, trail, C, wordmark, subtitle
  brand/logo/conductress-wordmark.svg      wordmark alone, outlined, with its chromatic fringe
  brand/logo/conductress-mark.svg          the keyhole C alone, 7 stripes, 100x100
  brand/logo/conductress-mark-32.svg       5-stripe cut for 32-64px
  brand/logo/conductress-mark-16.svg       3-stripe cut for 16px
  brand/logo/conductress-hero.png          1800x680
  brand/logo/conductress-mark-{512,180,32,16}.png
  brand/logo/favicon.ico                   16 + 32 + 48
  brand/pride/...                          the same set in the rainbow palette

Type is converted to outlines with fontTools so the SVGs render identically with
no font installed. Wordmark: Nimbus Sans Bold (URW's Helvetica). Subtitle:
DejaVu Sans Mono. Both are on every mainstream Linux; the paths to them are the
only machine-specific part of this file.

Geometry is the design's own: pointy-top hexagon R=46 about (50,50) in a 100-unit
box, keyhole = circle r=22 plus a channel (y 39..61) to the right edge. In the
lockup the lead hex is 126 units wide at (540,110).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_SANS = "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf"
FONT_MONO = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"

# ------------------------------------------------------------------ palette

SUNSET = {
    "cream": "#ffe08a", "amber": "#ffc94a", "orange": "#ff7a3d", "magenta": "#ff2d7e",
    "violet": "#6d3bd8", "midnight": "#150c26", "paper": "#fff6ea", "gold": "#c9a86e",
}
APPLE6 = ["#61bb46", "#fdb827", "#f5821f", "#e03a3e", "#963d97", "#009ddc"]
APPLE3 = ["#8fc63f", "#ea5a2e", "#4a6fbf"]

HEX_PATH = "M50 4 L89.8 27 L89.8 73 L50 96 L10.2 73 L10.2 27 Z"
C_PATH = "M89.8 27 L50 4 L10.2 27 L10.2 73 L50 96 L89.8 73 L89.8 61 L69.05 61 A22 22 0 1 1 69.05 39 L89.8 39 Z"
LEAD_HEX = "M540 52 L590.2 81 L590.2 139 L540 168 L489.8 139 L489.8 81 Z"

# ------------------------------------------------------------- text -> path

def text_path(text: str, font_path: str, size: float, cx: float, baseline: float, spacing: float) -> str:
    """Return an SVG path 'd' for text centred at cx on the given baseline."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    hmtx = font["hmtx"]
    s = size / font["head"].unitsPerEm
    names = [cmap[ord(ch)] for ch in text]
    advances = [hmtx[n][0] * s for n in names]
    width = sum(advances) + spacing * (len(text) - 1)
    x = cx - width / 2
    d = []
    for name, adv in zip(names, advances):
        pen = SVGPathPen(gs)
        gs[name].draw(TransformPen(pen, (s, 0, 0, -s, x, baseline)))
        cmd = pen.getCommands()
        if cmd:
            d.append(cmd)
        x += adv + spacing
    return " ".join(d)


# ----------------------------------------------------------------- pieces

def sunset_ramp(gid: str) -> str:
    return f"""    <linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="4" x2="0" y2="96">
      <stop offset="0%" stop-color="{SUNSET['cream']}"/>
      <stop offset="28%" stop-color="{SUNSET['amber']}"/>
      <stop offset="56%" stop-color="{SUNSET['orange']}"/>
      <stop offset="82%" stop-color="{SUNSET['magenta']}"/>
      <stop offset="100%" stop-color="{SUNSET['violet']}"/>
    </linearGradient>"""


def stripes(palette: str, count: int, fill_ref: str) -> str:
    """Stripe rects inside the 100-box, ready to be clipped by the C."""
    if palette == "sunset":
        plans = {
            7: [(4, 8.5), (17.5, 8.5), (31, 8.5), (44.5, 8.5), (58, 8.5), (71.5, 8.5), (85, 11)],
            5: [(4, 12), (22.5, 12), (41, 12), (59.5, 12), (78, 18)],
            3: [(4, 20), (35, 20), (66, 30)],
        }
        return "\n".join(
            f'      <rect x="8" y="{y}" width="84" height="{h}" fill="url(#{fill_ref})"/>' for y, h in plans[count]
        )
    if count == 6:  # gapped bands: 12.34 on a 15.33 pitch, last one solid to the base
        rows = [(4 + i * 15.333, 12.34 if i < 5 else 15.34, APPLE6[i]) for i in range(6)]
    elif count == 5:  # 6 touching bands for the 32px cut (gaps vanish at that size)
        rows = [(4 + i * 15.333, 15.34, APPLE6[i]) for i in range(6)]
    else:  # 3 merged bands for 16px
        rows = [(4 + i * 30.667, 30.67, APPLE3[i]) for i in range(3)]
    return "\n".join(f'      <rect x="8" y="{y:.2f}" width="84" height="{h}" fill="{c}"/>' for y, h, c in rows)


def mark_svg(palette: str, count: int) -> str:
    p = "s" if palette == "sunset" else "r"
    ramp = sunset_ramp(f"{p}-ramp") if palette == "sunset" else ""
    label = {"sunset": "sunset", "rainbow": "pride rainbow"}[palette]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Conductress mark: keyhole C, {label} palette, {count}-stripe cut. Generated by brand/tools/build.py -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100" role="img" aria-label="Conductress mark">
  <defs>
{ramp}
    <clipPath id="{p}-c"><path d="{C_PATH}"/></clipPath>
  </defs>
  <g clip-path="url(#{p}-c)">
{stripes(palette, count, f"{p}-ramp")}
  </g>
</svg>
"""


def scene_defs(palette: str, p: str) -> str:
    if palette == "sunset":
        S = SUNSET
        return f"""    <radialGradient id="{p}-sky" cx="50%" cy="38%" r="72%">
      <stop offset="0%" stop-color="#2a1140"/><stop offset="60%" stop-color="{S['midnight']}"/><stop offset="100%" stop-color="#07040d"/>
    </radialGradient>
    <linearGradient id="{p}-trail" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{S['violet']}"/><stop offset="55%" stop-color="{S['magenta']}"/><stop offset="100%" stop-color="{S['amber']}"/>
    </linearGradient>
    <linearGradient id="{p}-streak" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{S['violet']}" stop-opacity="0"/><stop offset="60%" stop-color="{S['magenta']}" stop-opacity="0.5"/><stop offset="100%" stop-color="{S['amber']}" stop-opacity="0.9"/>
    </linearGradient>
    <linearGradient id="{p}-grid" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{S['orange']}" stop-opacity="0.5"/><stop offset="100%" stop-color="{S['violet']}" stop-opacity="0.05"/>
    </linearGradient>
    <linearGradient id="{p}-rule" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{S['violet']}" stop-opacity="0"/><stop offset="25%" stop-color="{S['magenta']}"/><stop offset="75%" stop-color="{S['amber']}"/><stop offset="100%" stop-color="{S['amber']}" stop-opacity="0"/>
    </linearGradient>
{sunset_ramp(f"{p}-ramp")}"""
    bands = "".join(
        f'<stop offset="{i/6:.4f}" stop-color="{c}"/><stop offset="{(i+1)/6:.4f}" stop-color="{c}"/>' for i, c in enumerate(APPLE6)
    )
    return f"""    <radialGradient id="{p}-sky" cx="50%" cy="38%" r="72%">
      <stop offset="0%" stop-color="#1b1b28"/><stop offset="60%" stop-color="#0d0d16"/><stop offset="100%" stop-color="#050508"/>
    </radialGradient>
    <linearGradient id="{p}-trail" gradientUnits="userSpaceOnUse" x1="0" y1="52" x2="0" y2="168">{bands}</linearGradient>
    <linearGradient id="{p}-streak" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0"/><stop offset="100%" stop-color="#ffffff" stop-opacity="0.55"/>
    </linearGradient>
    <linearGradient id="{p}-grid" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.28"/><stop offset="100%" stop-color="#ffffff" stop-opacity="0.03"/>
    </linearGradient>
    <linearGradient id="{p}-rule" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0"/><stop offset="50%" stop-color="#d8d8e6"/><stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>"""


def trail_paths() -> str:
    out = []
    for i in range(10):
        cx = 370 + 17 * i
        w = 1.4 + 0.2 * i
        op = [0.17, 0.24, 0.32, 0.40, 0.48, 0.57, 0.66, 0.75, 0.84, 0.92][i]
        out.append(
            f'    <path d="M{cx} 52 L{cx+50.2} 81 L{cx+50.2} 139 L{cx} 168 L{cx-50.2} 139 L{cx-50.2} 81 Z" stroke-width="{w:.1f}" opacity="{op}"/>'
        )
    return "\n".join(out)


STREAKS = """    <line x1="150" y1="72" x2="372" y2="72" stroke-width="2"/>
    <line x1="118" y1="96" x2="356" y2="96" stroke-width="3.4"/>
    <line x1="166" y1="120" x2="366" y2="120" stroke-width="1.6"/>
    <line x1="104" y1="144" x2="350" y2="144" stroke-width="4.2"/>
    <line x1="182" y1="164" x2="360" y2="164" stroke-width="1.4"/>"""

GRID = """    <line x1="60" y1="196" x2="840" y2="196" stroke="url(#{p}-rule)" stroke-width="1.6"/>
    <g stroke="url(#{p}-grid)" stroke-width="1.1" fill="none">
      <line x1="450" y1="196" x2="-140" y2="330"/><line x1="450" y1="196" x2="60" y2="330"/>
      <line x1="450" y1="196" x2="248" y2="330"/><line x1="450" y1="196" x2="380" y2="330"/>
      <line x1="450" y1="196" x2="520" y2="330"/><line x1="450" y1="196" x2="652" y2="330"/>
      <line x1="450" y1="196" x2="840" y2="330"/><line x1="450" y1="196" x2="1040" y2="330"/>
    </g>"""


def wordmark_group(palette: str, p: str, word_d: str) -> str:
    if palette == "sunset":
        left, right, op = SUNSET["magenta"], SUNSET["amber"], 0.75
    else:
        left, right, op = APPLE6[3], APPLE6[5], 0.7
    return f"""  <g id="wordmark">
    <path d="{word_d}" transform="translate(2 0)" fill="{left}" opacity="{op}"/>
    <path d="{word_d}" transform="translate(-2 0)" fill="{right}" opacity="{op}"/>
    <path d="{word_d}" fill="{SUNSET['paper']}" filter="url(#{p}-soft)"/>
  </g>"""


def subtitle_group(palette: str, p: str, sub_d: str) -> str:
    fill = SUNSET["gold"] if palette == "sunset" else "#9a9ab0"
    return f"""  <g id="subtitle">
    <line x1="170" y1="276" x2="730" y2="276" stroke="url(#{p}-rule)" stroke-width="2.2"/>
    <line x1="230" y1="284" x2="670" y2="284" stroke="url(#{p}-rule)" stroke-width="1.1"/>
    <path d="{sub_d}" fill="{fill}"/>
  </g>"""


def not_lead_clip() -> str:
    """Everything except the lead hexagon, as a UNION of six outward half-planes.

    A single evenodd path would be shorter, but cairosvg ignores clip-rule, and a
    mask vanished entirely in earlier renders. Six same-orientation polygons under
    the default nonzero rule work in browsers and cairosvg alike.
    """
    pts = [(540, 52), (590.2, 81), (590.2, 139), (540, 168), (489.8, 139), (489.8, 81)]
    cx, cy, L = 540.0, 110.0, 5000.0
    polys = []
    for i in range(6):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 6]
        dx, dy = bx - ax, by - ay
        ln = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / ln, dy / ln
        # outward normal: the one pointing away from the centre
        nx, ny = dy, -dx
        mx, my = (ax + bx) / 2 - cx, (ay + by) / 2 - cy
        if nx * mx + ny * my < 0:
            nx, ny = -nx, -ny
        a2 = (ax - dx * L, ay - dy * L)
        b2 = (bx + dx * L, by + dy * L)
        quad = [a2, b2, (b2[0] + nx * L, b2[1] + ny * L), (a2[0] + nx * L, a2[1] + ny * L)]
        polys.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in quad) + '"/>')
    return "".join(polys)


STARS = [
    (736, 110, 1.0, .61), (729, 162, 1.6, .71), (636, 25, 1.4, .62), (64, 157, 0.8, .70), (660, 153, 0.9, .46),
    (726, 97, 1.0, .75), (202, 27, 1.2, .81), (87, 137, 1.6, .43), (714, 151, 1.0, .56), (100, 43, 0.8, .58),
    (849, 113, 1.1, .77), (813, 41, 1.1, .61), (47, 34, 1.6, .50), (694, 162, 0.9, .79), (806, 16, 0.8, .38),
    (70, 34, 1.5, .55), (172, 161, 1.5, .36), (842, 49, 1.4, .47), (809, 123, 1.3, .84), (835, 150, 1.3, .53),
    (692, 71, 1.3, .54), (143, 64, 1.1, .72), (157, 51, 1.0, .36), (81, 29, 1.7, .68), (662, 128, 0.8, .72),
    (145, 164, 1.0, .81), (271, 91, 1.6, .38), (817, 94, 1.6, .76), (799, 171, 1.7, .50), (160, 22, 1.6, .66),
    (784, 66, 1.0, .37), (204, 106, 1.1, .83), (133, 102, 1.1, .78), (638, 141, 1.5, .67),
]
MOUNTAINS = (
    "M0 196 L38 186 L74 191 L112 178 L146 189 L184 181 L221 190 L258 183 L292 191 L326 187 L360 194 L385 196 Z",
    "M515 196 L540 194 L568 189 L606 183 L640 190 L676 180 L712 189 L748 176 L786 188 L820 182 L858 190 L900 185 L900 196 Z",
)


def scenery() -> str:
    """Stars either side of the mark, mountain silhouettes on the horizon, black ground.
    Identical to the sign-on's resting frame (brand/motion), which is the rule."""
    stars = "\n".join(f'    <circle cx="{x}" cy="{y}" r="{r}" opacity="{o:.2f}"/>' for x, y, r, o in STARS)
    mts = "\n".join(f'    <path d="{d}"/>' for d in MOUNTAINS)
    return f"""  <g id="stars" fill="{SUNSET['paper']}">
{stars}
  </g>
  <g id="mountains" fill="#07040c">
{mts}
  </g>
  <rect id="ground" x="0" y="196" width="900" height="144" fill="#000"/>
"""


def lockup_svg(palette: str, opaque: bool, word_d: str, sub_d: str) -> str:
    p = ("s" if palette == "sunset" else "r") + ("h" if opaque else "l")
    ramp_ref = f"{p}-ramp"
    kind = "hero" if opaque else "lockup"
    background = (
        f"""  <g id="background">
    <rect width="900" height="340" fill="url(#{p}-sky)"/>
    <circle cx="450" cy="110" r="150" fill="{SUNSET['orange'] if palette == 'sunset' else '#ffffff'}" opacity="{0.07 if palette == 'sunset' else 0.05}"/>
  </g>
{scenery()}  <g id="horizon" opacity="0.7">
{GRID.format(p=p)}
  </g>
"""
        if opaque
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Conductress {kind}, {palette} palette. Generated by brand/tools/build.py; edit that, not this.
     Type is outlined (Nimbus Sans Bold / DejaVu Sans Mono); the trail is clipped out of the lead
     hexagon so the keyhole and stripe gaps show whatever is behind the logo. -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 340" width="900" height="340" role="img" aria-label="Conductress">
  <defs>
{scene_defs(palette, p)}
    <filter id="{p}-bloom" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="7" result="b1"/><feMerge><feMergeNode in="b1"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="{p}-soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="3" result="b2"/><feMerge><feMergeNode in="b2"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <!-- everything except the lead hexagon: the trail is drawn through this so nothing sits behind the C -->
    <clipPath id="{p}-notlead">{not_lead_clip()}</clipPath>
    <clipPath id="{p}-c"><path d="{C_PATH}"/></clipPath>
    <symbol id="{p}-cmark" viewBox="0 0 100 100">
      <g clip-path="url(#{p}-c)">
{stripes(palette, 7 if palette == "sunset" else 6, ramp_ref)}
      </g>
    </symbol>
  </defs>

{background}  <g id="speedstreaks" stroke="url(#{p}-streak)" stroke-linecap="round">
{STREAKS}
  </g>

  <g id="trail" fill="none" stroke="url(#{p}-trail)" stroke-linejoin="round" clip-path="url(#{p}-notlead)">
{trail_paths()}
  </g>

  <g id="lead" filter="url(#{p}-bloom)">
    <use href="#{p}-cmark" x="477" y="47" width="126" height="126"/>
  </g>

{wordmark_group(palette, p, word_d)}

{subtitle_group(palette, p, sub_d)}
</svg>
"""


def wordmark_svg(palette: str, word_d: str) -> str:
    p = ("s" if palette == "sunset" else "r") + "w"
    # wordmark alone: shift the 900x340 coordinates into a tight 640x80 box
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Conductress wordmark, outlined (Nimbus Sans Bold 54/13). Generated by brand/tools/build.py -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="130 200 640 72" width="640" height="72" role="img" aria-label="CONDUCTRESS">
  <defs>
    <filter id="{p}-soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="3" result="b2"/><feMerge><feMergeNode in="b2"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
{wordmark_group(palette, p, word_d)}
</svg>
"""


def palette_svg() -> str:
    """Swatch strip for the guide: the sunset roles on top, the six pride bands below."""
    sunset = [("cream", "#ffe08a"), ("amber", "#ffc94a"), ("orange", "#ff7a3d"), ("magenta", "#ff2d7e"),
              ("violet", "#6d3bd8"), ("gold", "#c9a86e"), ("midnight", "#150c26")]
    pride = list(zip(["green", "yellow", "orange", "red", "purple", "blue"], APPLE6))
    w = 100
    rows = []
    for r, row in enumerate((sunset, pride)):
        for i, (name, hexv) in enumerate(row):
            x, y = i * w, r * 92
            rows.append(f'  <rect x="{x}" y="{y}" width="{w}" height="56" fill="{hexv}"/>')
            rows.append(f'  <path d="{text_path(name, FONT_MONO, 11, x + w / 2, y + 70, 0)}" fill="#c9bfe6"/>')
            rows.append(f'  <path d="{text_path(hexv, FONT_MONO, 11, x + w / 2, y + 84, 0)}" fill="#8f83b5"/>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Conductress palette. Generated by brand/tools/build.py -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 184" width="700" height="184" role="img" aria-label="Conductress colour palette">
  <rect width="700" height="184" fill="#0b0812"/>
{chr(10).join(rows)}
</svg>
"""


# ------------------------------------------------------------------ build

def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(ROOT.parent))


def png(svg_path: Path, out: Path, width: int) -> None:
    import cairosvg

    cairosvg.svg2png(url=str(svg_path), write_to=str(out), output_width=width)
    print("wrote", out.relative_to(ROOT.parent))


def ico(mark16: Path, mark32: Path, out: Path) -> None:
    import cairosvg
    from PIL import Image

    frames = []
    for src, size in ((mark16, 16), (mark32, 32), (mark32, 48)):
        buf = io.BytesIO(cairosvg.svg2png(url=str(src), output_width=size, output_height=size))
        frames.append(Image.open(buf).convert("RGBA"))
    frames[0].save(out, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)], append_images=frames[1:])
    print("wrote", out.relative_to(ROOT.parent))


def main() -> None:
    for f in (FONT_SANS, FONT_MONO):
        if not os.path.exists(f):
            sys.exit(f"font not found: {f} -- edit FONT_SANS/FONT_MONO at the top of this file")
    word_d = text_path("CONDUCTRESS", FONT_SANS, 54, 450, 252, 13)
    sub_d = text_path("ONLY DATA IS REAL", FONT_MONO, 15, 450, 312, 8)
    write(ROOT / "palette.svg", palette_svg())

    for palette, folder in (("sunset", "logo"), ("rainbow", "pride")):
        d = ROOT / folder
        suffix = "" if palette == "sunset" else "-pride"
        write(d / f"conductress-hero{suffix}.svg", lockup_svg(palette, True, word_d, sub_d))
        write(d / f"conductress-lockup{suffix}.svg", lockup_svg(palette, False, word_d, sub_d))
        write(d / f"conductress-wordmark{suffix}.svg", wordmark_svg(palette, word_d))
        big, mid, small = (7, 5, 3) if palette == "sunset" else (6, 5, 3)
        write(d / f"conductress-mark{suffix}.svg", mark_svg(palette, big))
        write(d / f"conductress-mark{suffix}-32.svg", mark_svg(palette, mid))
        write(d / f"conductress-mark{suffix}-16.svg", mark_svg(palette, small))

        png(d / f"conductress-hero{suffix}.svg", d / f"conductress-hero{suffix}.png", 1800)
        png(d / f"conductress-mark{suffix}.svg", d / f"conductress-mark{suffix}-512.png", 512)
        png(d / f"conductress-mark{suffix}.svg", d / f"conductress-mark{suffix}-180.png", 180)
        png(d / f"conductress-mark{suffix}-32.svg", d / f"conductress-mark{suffix}-32.png", 32)
        png(d / f"conductress-mark{suffix}-16.svg", d / f"conductress-mark{suffix}-16.png", 16)
        ico(d / f"conductress-mark{suffix}-16.svg", d / f"conductress-mark{suffix}-32.svg", d / f"favicon{suffix}.ico")


if __name__ == "__main__":
    main()
