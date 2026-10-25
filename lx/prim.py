import pymupdf as fitz, hashlib

def col(c):
    return None if c is None else tuple(round(x, 2) for x in c)

def prim_hash(d, size_q=0.3, grid=3):
    """Translation-invariant, clip-independent shape signature of one drawing:
    paint style + quantised size + path-op sequence + coarse control-point layout."""
    r = d['rect']; w, h = r.width, r.height
    kinds = []; cells = []
    for it in d['items']:
        kinds.append(it[0])
        for a in it[1:]:
            pts = []
            if hasattr(a, 'x'): pts = [a]
            elif hasattr(a, 'x0'): pts = [fitz.Point(a.x0, a.y0), fitz.Point(a.x1, a.y1)]
            for q in pts:
                cx = min(grid - 1, int((q.x - r.x0) / w * grid)) if w > 1e-6 else 0
                cy = min(grid - 1, int((q.y - r.y0) / h * grid)) if h > 1e-6 else 0
                cells.append(cx * grid + cy)
    key = (col(d.get('fill')), col(d.get('color')), round(d.get('width') or 0, 1),
           round(w / size_q), round(h / size_q), tuple(kinds), tuple(cells[:24]))
    return hashlib.md5(repr(key).encode()).hexdigest()[:10]
