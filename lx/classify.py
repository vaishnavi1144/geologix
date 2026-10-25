"""Classify visible painted runs against the document's own legend (colour + vector pattern fingerprint)."""
import numpy as np, collections, math
from prim import col

def build_index(entries):
    """entries: legend dicts (name, bg, prims Counter, drawings). Returns lookup structures."""
    df = collections.Counter()
    for e in entries:
        for h in e['prims']: df[h] += 1
    N = max(1, len(entries))
    idf = {h: math.log((1 + N) / (1 + c)) + 0.2 for h, c in df.items()}
    return idf

def colour_close(a, b, tol=0.025):
    return a is not None and b is not None and all(abs(x - y) <= tol for x, y in zip(a, b))

def row_runs(cv, xa, xb, top_row, bot_row):
    """Dominant colour id per row inside the column; returns list of (r0, r1, colour_id)."""
    ca = int((xa - cv.x0) / cv.rx); cb = int(np.ceil((xb - cv.x0) / cv.rx))
    blk = cv.cid[top_row:bot_row, ca:cb]
    dom = np.zeros(blk.shape[0], dtype=np.int16)
    for i in range(blk.shape[0]):
        v = np.bincount(blk[i], minlength=len(cv.colours))
        dom[i] = int(np.argmax(v))
    runs = []; s = 0
    for i in range(1, len(dom) + 1):
        if i == len(dom) or dom[i] != dom[s]:
            runs.append([top_row + s, top_row + i, int(dom[s])]); s = i
    return runs

def score_run(prims_in_run, cands, idf):
    """Vote: each visible pattern primitive whose hash occurs in a candidate swatch votes with idf weight."""
    sc = {e['name'] + '|' + str(id(e)): 0.0 for e in cands}
    for _, d in prims_in_run:
        h = d['h']
        for e in cands:
            c = e['prims'].get(h)
            if c: sc[e['name'] + '|' + str(id(e))] += idf.get(h, 0.2)
    return sc
