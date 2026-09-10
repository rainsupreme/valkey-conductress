#!/usr/bin/env python3
"""Conductress keyhole C-mark for the terminal.

BRIEF: terminal cut of conductress-logo-g-c-mark.svg (sunset) and
conductress-logo-g-c-mark-rainbow.svg (retro Apple rainbow, pride month).
The speed trail is dropped: 1px hexagon outlines at 12-44px are staircases,
not a trail. The C alone reduces well because it is a filled silhouette.

Geometry is the SVG's own: pointy-top hexagon R=46 about (50,50); keyhole =
circle r=22 at centre plus a channel (y 39..61) out to the right edge.

Rendering:
  * Pixel grid N x N (N even). Each terminal cell is 1 col x 2 px using
    half-block glyphs (U+2580 upper, U+2584 lower, U+2588 full), so pixels
    are square in a normal 1:2 terminal cell. Background is left alone so
    the mark sits on whatever the terminal is. --quadrant instead packs a
    2x2 grid of half-width subpixels per cell using the Block Elements
    quadrant glyphs (U+2596-259F) -- diagonals get roughly twice the
    horizontal resolution and read as a slope instead of a staircase, at
    the cost of one averaged colour per cell instead of a true top/bottom
    split (only matters where a cell straddles a stripe boundary).
  * Stripes are laid out in WHOLE pixel rows (not sampled), so gaps are
    always crisp. Stripe count drops with size: 7 -> 5 -> 3 (sunset),
    6 gapped -> 6 touching (rainbow). 12px was cut: too small to read.
  * Sunset colour is the whole-mark gradient sampled per pixel row
    (#ffe08a -> #ffc94a -> #ff7a3d -> #ff2d7e -> #6d3bd8), same stops as
    the SVG's userSpaceOnUse ramp.
  * --colors 256 quantises to the xterm 6x6x6 cube for older terminals.
  * --ascii uses '#' only (7-bit safe), 2 chars per pixel horizontally.
  * --html writes a preview page with the same cells as <span>s.

Usage:
  ./conductress-logo-ascii.py                    # both palettes, 44 + 24px lockups
  ./conductress-logo-ascii.py --palette sunset --size 24
  ./conductress-logo-ascii.py --palette rainbow --mark-only
  ./conductress-logo-ascii.py --colors 256 > logo.ansi
  ./conductress-logo-ascii.py --ascii
"""
import argparse
import math
import sys

# ----------------------------------------------------------------- palette

SUNSET_STOPS = [
    (0.00, "#ffe08a"),
    (0.28, "#ffc94a"),
    (0.56, "#ff7a3d"),
    (0.82, "#ff2d7e"),
    (1.00, "#6d3bd8"),
]
APPLE6 = ["#61bb46", "#fdb827", "#f5821f", "#e03a3e", "#963d97", "#009ddc"]
APPLE3 = ["#8fc63f", "#ea5a2e", "#4a6fbf"]  # pairs merged at their midpoints
CREAM = "#ffe08a"
DIM = "#6d5f92"
BG = "#150c26"

UPPER, LOWER, FULL = "\u2580", "\u2584", "\u2588"


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def lerp_stops(stops, t):
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            a, b = rgb(c0), rgb(c1)
            return tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3))
    return rgb(stops[-1][1])


def to_256(c):
    """Nearest xterm 6x6x6 cube index (ignores the grey ramp; fine for these hues)."""
    def q(v):
        return 0 if v < 48 else 1 if v < 115 else (v - 35) // 40
    r, g, b = (q(v) for v in c)
    return 16 + 36 * r + 6 * g + b


# ---------------------------------------------------------------- geometry

HEX = [(50, 4), (89.8, 27), (89.8, 73), (50, 96), (10.2, 73), (10.2, 27)]


def in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def in_c(x, y, half_channel=11.0):
    """Inside the C: inside the hexagon and outside the keyhole.

    half_channel is the SVG's 11 (y 39..61). The 12px cut widens it to 17 so the
    mouth spans four whole pixel rows; at 11 it is two rows and reads as a donut.
    """
    if not in_poly(x, y, HEX):
        return False
    if (x - 50) ** 2 + (y - 50) ** 2 <= 22 ** 2:
        return False
    if x >= 50 and 50 - half_channel <= y <= 50 + half_channel:
        return False
    return True


def shape_mask(n, cols=None, ss=4):
    """cols x n boolean grid: pixel filled if >=50% of ss*ss samples are inside the C.

    cols defaults to n (square pixels, half-block mode). Quadrant mode passes 2n
    for half-width subpixels.
    """
    cols = cols or n
    sx_scale = 100.0 / cols
    sy_scale = 100.0 / n
    half_channel = 11.0 if n >= 20 else 17.0
    mask = []
    for py in range(n):
        row = []
        for px in range(cols):
            hits = 0
            for sy in range(ss):
                for sx in range(ss):
                    x = (px + (sx + 0.5) / ss) * sx_scale
                    y = (py + (sy + 0.5) / ss) * sy_scale
                    if in_c(x, y, half_channel):
                        hits += 1
            row.append(hits * 2 >= ss * ss)
        mask.append(row)
    return mask


# ----------------------------------------------------------------- stripes

def stripe_plan(n, palette):
    """Choose stripe count and gap (in pixel rows) for a given pixel size."""
    if palette == "sunset":
        if n >= 40:
            return 7, 2
        if n >= 20:
            return 5, 1
        return 3, 1
    # rainbow
    if n >= 40:
        return 6, 1
    if n >= 20:
        return 6, 0
    return 3, 0


def row_colors(n, palette):
    """Per pixel row: RGB tuple or None (gap / outside the 4..96 band)."""
    r0 = round(0.04 * n)
    r1 = round(0.96 * n)
    total = r1 - r0
    count, gap = stripe_plan(n, palette)
    stripe_h = (total - gap * (count - 1)) / count
    rows = [None] * n
    for i in range(count):
        a = r0 + round(i * (stripe_h + gap))
        b = r0 + round(i * (stripe_h + gap) + stripe_h)
        if i == count - 1:
            b = r1  # bottom stripe absorbs rounding, like the SVG's heavier base
        for r in range(a, b):
            if palette == "sunset":
                rows[r] = lerp_stops(SUNSET_STOPS, (r + 0.5 - r0) / total)
            else:
                bands = APPLE6 if count == 6 else APPLE3
                rows[r] = rgb(bands[i])
    return rows


def pixels(n, palette, cols=None):
    """rows(n) x cols grid of RGB or None. cols defaults to n."""
    cols = cols or n
    mask = shape_mask(n, cols=cols)
    rows = row_colors(n, palette)
    return [[rows[y] if mask[y][x] else None for x in range(cols)] for y in range(n)]


# ---------------------------------------------------------------- emitters

class Ansi:
    def __init__(self, depth=24):
        self.depth = depth

    def fg(self, c):
        if self.depth == 256:
            return "\x1b[38;5;%dm" % to_256(c)
        return "\x1b[38;2;%d;%d;%dm" % c

    def bg(self, c):
        if self.depth == 256:
            return "\x1b[48;5;%dm" % to_256(c)
        return "\x1b[48;2;%d;%d;%dm" % c

    reset = "\x1b[0m"

    def cell(self, glyph, fg=None, bg=None):
        if fg is None and bg is None:
            return glyph
        return (self.fg(fg) if fg else "") + (self.bg(bg) if bg else "") + glyph + self.reset

    def text(self, s, fg=None, bold=False, dim=False):
        pre = ("\x1b[1m" if bold else "") + ("\x1b[2m" if dim else "") + (self.fg(fg) if fg else "")
        return pre + s + self.reset


class Html:
    reset = ""

    def cell(self, glyph, fg=None, bg=None):
        if fg is None and bg is None:
            return glyph
        style = ""
        if fg:
            style += "color:#%02x%02x%02x;" % fg
        if bg:
            style += "background:#%02x%02x%02x;" % bg
        return '<span style="%s">%s</span>' % (style, glyph)

    def text(self, s, fg=None, bold=False, dim=False):
        style = ""
        if fg:
            style += "color:#%02x%02x%02x;" % fg
        if bold:
            style += "font-weight:bold;"
        if dim:
            style += "opacity:.7;"
        return '<span style="%s">%s</span>' % (style, s)


def render_blocks(px, out):
    """Half-block rendering: one text row per two pixel rows."""
    n = len(px)
    lines = []
    for cy in range(0, n, 2):
        top, bot = px[cy], px[cy + 1] if cy + 1 < n else [None] * n
        cells = []
        for x in range(n):
            t, b = top[x], bot[x]
            if t and b:
                cells.append(out.cell(FULL, fg=t) if t == b else out.cell(UPPER, fg=t, bg=b))
            elif t:
                cells.append(out.cell(UPPER, fg=t))
            elif b:
                cells.append(out.cell(LOWER, fg=b))
            else:
                cells.append(" ")
        lines.append("".join(cells).rstrip())
    return lines


def render_ascii(px, out, glyph="#"):
    """7-bit rendering: one text row per pixel row, two chars per pixel."""
    lines = []
    for row in px:
        cells = []
        for c in row:
            cells.append(out.cell(glyph * 2, fg=c) if c else "  ")
        lines.append("".join(cells).rstrip())
    return lines


# Quadrant glyphs, TL/TR/BL/BR bit order (bit0=TL, bit1=TR, bit2=BL, bit3=BR).
# All 16 exist in Block Elements/Symbols for Legacy Computing except the two
# that coincide with the half-blocks already in this range (0011->LOWER HALF,
# 1100->UPPER HALF are handled as special cases below for font compatibility).
_QUAD = {
    0b0000: " ",
    0b0001: "\u2598",  # ▘ TL
    0b0010: "\u259d",  # ▝ TR
    0b0011: "\u2580",  # ▀ top half
    0b0100: "\u2596",  # ▖ BL
    0b0101: "\u258c",  # ▌ left half
    0b0110: "\u259e",  # ▞ TR+BL
    0b0111: "\u259b",  # ▛ TL+TR+BL
    0b1000: "\u2597",  # ▗ BR
    0b1001: "\u259a",  # ▚ TL+BR
    0b1010: "\u2590",  # ▐ right half
    0b1011: "\u259c",  # ▜ TL+TR+BR
    0b1100: "\u2584",  # ▄ bottom half
    0b1101: "\u2599",  # ▙ TL+BL+BR
    0b1110: "\u259f",  # ▟ TR+BL+BR
    0b1111: "\u2588",  # █ full
}


def render_quadrants(px, out):
    """2x2 subpixels per cell: half rows x half-width cols -> one glyph.

    One foreground colour per cell (quadrant glyphs have no bg channel in most
    fonts' metrics), chosen as the average of the filled subpixels' colours so
    a cell straddling a gradient boundary blends rather than snapping.
    """
    n = len(px)
    cols = len(px[0]) if n else 0
    lines = []
    for cy in range(0, n, 2):
        r0 = px[cy]
        r1 = px[cy + 1] if cy + 1 < n else [None] * cols
        cells = []
        for cx in range(0, cols, 2):
            quad = [
                r0[cx] if cx < cols else None,
                r0[cx + 1] if cx + 1 < cols else None,
                r1[cx] if cx < cols else None,
                r1[cx + 1] if cx + 1 < cols else None,
            ]
            bits = sum((1 << i) for i, c in enumerate(quad) if c)
            glyph = _QUAD[bits]
            if bits == 0:
                cells.append(" ")
                continue
            filled = [c for c in quad if c]
            avg = tuple(round(sum(ch[i] for ch in filled) / len(filled)) for i in range(3))
            cells.append(out.cell(glyph, fg=avg))
        lines.append("".join(cells).rstrip())
    return lines


def wordmark_colors(palette, count):
    if palette == "sunset":
        return [lerp_stops(SUNSET_STOPS, i / max(1, count - 1)) for i in range(count)]
    return [rgb(APPLE6[(i * 6) // count]) for i in range(count)]


def lockup(lines, palette, out, gap="   "):
    """Mark on the left, spaced wordmark and subtitle on the right, vertically centred."""
    width = max(visible_len(l) for l in lines)
    word = "CONDUCTRESS"
    cols = wordmark_colors(palette, len(word))
    spaced = " ".join(out.text(ch, fg=cols[i], bold=True) for i, ch in enumerate(word))
    rule = out.text("\u2500" * (2 * len(word) - 1), fg=rgb(DIM))
    sub = out.text("ONLY DATA IS REAL", fg=rgb(DIM))
    mid = len(lines) // 2
    right = {mid - 1: spaced, mid: rule, mid + 1: sub}
    result = []
    for i, line in enumerate(lines):
        pad = " " * (width - visible_len(line))
        result.append(line + pad + (gap + right[i] if i in right else ""))
    return result


def visible_len(s):
    """Length ignoring ANSI SGR sequences and HTML tags."""
    import re
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return len(s)


# -------------------------------------------------------------------- main

def build(args, out):
    sizes = [args.size] if args.size else [44, 24]
    palettes = [args.palette] if args.palette else ["sunset", "rainbow"]
    mode = "ascii" if args.ascii else ("quadrant" if args.quadrant else "half-block")
    blocks = []
    for pal in palettes:
        for n in sizes:
            if args.quadrant:
                px = pixels(n, pal, cols=2 * n)
                lines = render_quadrants(px, out)
            else:
                px = pixels(n, pal)
                lines = render_ascii(px, out) if args.ascii else render_blocks(px, out)
            if not args.mark_only:
                lines = lockup(lines, pal, out)
            label = out.text("%s  %dpx  %s" % (pal.upper(), n, mode), dim=True)
            blocks.append((label, lines))
    return blocks


def main():
    ap = argparse.ArgumentParser(description="Conductress C-mark for the terminal")
    ap.add_argument("--palette", choices=["sunset", "rainbow"])
    ap.add_argument("--size", type=int, help="pixel size (even). Default: 44 and 24")
    ap.add_argument("--colors", type=int, choices=[24, 256], default=24, help="24-bit (default) or 256")
    ap.add_argument("--ascii", action="store_true", help="7-bit glyphs only, one row per pixel")
    ap.add_argument(
        "--quadrant",
        action="store_true",
        help="2x2 subpixels per cell (Block Elements quadrant glyphs) for crisper diagonals; "
        "one colour per cell, needs a font with U+2596-259F (most do)",
    )
    ap.add_argument("--mark-only", action="store_true", help="omit the wordmark and subtitle (default: lockup)")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--html", metavar="FILE", help="write an HTML preview instead of ANSI")
    args = ap.parse_args()
    if args.size and args.size % 2:
        ap.error("--size must be even")
    if args.ascii and args.quadrant:
        ap.error("--ascii and --quadrant are mutually exclusive")

    if args.html:
        out = Html()
        blocks = build(args, out)
        body = []
        for label, lines in blocks:
            if not args.no_labels:
                body.append(label)
            body.extend(lines)
            body.append("")
        html = (
            "<!doctype html><meta charset='utf-8'><style>"
            "body{background:%s;margin:24px}"
            "pre{font-family:'DejaVu Sans Mono',Menlo,Consolas,monospace;font-size:16px;"
            "line-height:1;color:#c9bfe6;margin:0}"
            "</style><pre>%s</pre>" % (BG, "\n".join(body))
        )
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print("wrote", args.html)
        return

    out = Ansi(args.colors)
    for label, lines in build(args, out):
        if not args.no_labels:
            sys.stdout.write(label + "\n")
        sys.stdout.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
