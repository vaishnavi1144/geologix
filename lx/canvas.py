"""2-D painter's-algorithm canvas built from vector fills (no rasterised page, no anti-aliasing)."""
import numpy as np, collections
from prim import col

def is_rect_fill(d):
    it = d['items']
    if len(it) == 1 and it[0][0] == 're': return True
    if 4 <= len(it) <= 5 and all(i[0] in ('l', 're') for i in it):
        xs = set(); ys = set()
        for i in it:
            for a in i[1:]:
                if hasattr(a, 'x'): xs.add(round(a.x, 1)); ys.add(round(a.y, 1))
        return len(xs) <= 2 and len(ys) <= 2
    return False

class Canvas:
    def __init__(self, drs, x0, y0, x1, y1, rx=1.0, ry=0.5, min_paint_w=0.4):
        self.x0, self.y0, self.rx, self.ry = x0, y0, rx, ry
        self.W = int((x1 - x0) / rx) + 1; self.H = int((y1 - y0) / ry) + 1
        self.cid = np.zeros((self.H, self.W), dtype=np.int16)      # colour id (0 = unpainted)
        self.seq = np.zeros((self.H, self.W), dtype=np.int32)      # z-order of last paint
        self.colours = [None]; self._cmap = {}
        self.prims = []                                            # non-rect drawings
        for s, d in enumerate(drs, 1):
            e = d['eff']
            if e.x1 < x0 or e.x0 > x1 or e.y1 < y0 or e.y0 > y1: continue
            if d['type'] in ('f', 'fs') and d.get('fill') and is_rect_fill(d) and e.height >= 0.4 and e.width >= min_paint_w:
                c = col(d['fill'])
                if c not in self._cmap:
                    self._cmap[c] = len(self.colours); self.colours.append(c)
                a, b, c0, c1 = self._idx(e)
                self.cid[a:b, c0:c1] = self._cmap[c]; self.seq[a:b, c0:c1] = s
            else:
                self.prims.append((s, d))
    def _idx(self, e):
        a = max(0, int(round((e.y0 - self.y0) / self.ry))); b = min(self.H, int(round((e.y1 - self.y0) / self.ry)))
        c0 = max(0, int(round((e.x0 - self.x0) / self.rx))); c1 = min(self.W, int(round((e.x1 - self.x0) / self.rx)))
        return a, max(b, a + 1), c0, max(c1, c0 + 1)
    def y_of_row(self, r): return self.y0 + r * self.ry
    def x_of_col(self, c): return self.x0 + c * self.rx
    def visible_prims(self, xa, xb):
        """prims inside [xa,xb] that were not painted over later (by a rect fill)."""
        out = []
        for s, d in self.prims:
            e = d['eff']
            if e.x0 < xa - 1 or e.x1 > xb + 1: continue
            a, b, c0, c1 = self._idx(e)
            if a >= self.H: continue
            ca = (a + b) // 2; cc = (c0 + c1) // 2
            if s > self.seq[min(ca, self.H - 1), min(cc, self.W - 1)]:
                out.append((s, d))
        return out
