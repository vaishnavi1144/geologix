"""Load page drawings with clip paths applied (effective visible rect)."""
import pymupdf as fitz
from prim import prim_hash

def load_drawings(page):
    ex = page.get_drawings(extended=True)
    stack = []                       # (level, scissor Rect)
    out = []
    for d in ex:
        lvl = d.get('level', 0)
        while stack and stack[-1][0] >= lvl and d['type'] != 'clip':
            stack.pop()
        if d['type'] == 'clip':
            while stack and stack[-1][0] >= lvl: stack.pop()
            stack.append((lvl, d['scissor'])); continue
        r = fitz.Rect(d['rect'])
        for _, sc in stack:
            r = r & sc if not r.is_empty or True else r
            if r.is_empty and d['rect'].width == 0 or d['rect'].height == 0:
                # degenerate (line) rect: intersect manually
                r = fitz.Rect(max(d['rect'].x0, sc.x0), max(d['rect'].y0, sc.y0),
                              min(d['rect'].x1, sc.x1), min(d['rect'].y1, sc.y1))
        if r.x1 < r.x0 or r.y1 < r.y0: continue      # fully clipped away
        d['eff'] = r
        d['clip'] = stack[-1][1] if stack else None
        d['h'] = prim_hash(d)
        out.append(d)
    return out
