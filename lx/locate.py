import numpy as np, collections
from prim import col

def column_extents(cv, cal, words, min_w=8, gray_tol=0.04, drs=None):
    """Find vertical tracks that carry coloured fills. Returns ranked candidates."""
    top_r = max(0, int((cal['top'] - cv.y0) / cv.ry)); bot_r = min(cv.H, int((cal['bottom'] - cv.y0) / cv.ry))
    sub = cv.cid[top_r:bot_r]
    # colour ids that are "ink" (not white, not neutral grey/black used for grid/curves)
    def chroma(c): return None if c is None else max(c) - min(c)
    ok = np.zeros(len(cv.colours), dtype=bool)
    for i, c in enumerate(cv.colours):
        if c is not None and not all(v > 0.97 for v in c): ok[i] = True
    cov = ok[sub].mean(axis=0) if sub.size else np.array([])
    peak = cov.max() if cov.size else 0
    if peak < 0.02:
        # Fallback to search entire canvas height if depth slice is sparse
        sub = cv.cid
        cov = ok[sub].mean(axis=0) if sub.size else np.array([])
        peak = cov.max() if cov.size else 0
        if peak < 0.02: return []
    active = cov > max(0.04, 0.25 * peak)
    runs = []; s = None
    for i, a in enumerate(active):
        if a and s is None: s = i
        elif not a and s is not None: runs.append((s, i)); s = None
    if s is not None: runs.append((s, len(active)))
    cands = []
    hdr = [w for w in words if (cal['top'] - 1000 < w[1] < cal['top'] + 10) or (cal['bottom'] - 10 < w[1] < cal['bottom'] + 300)]
    hdr = [w for w in hdr if w[4].upper() not in ('SYMBOLS', 'SYMBOL')]
    for a, b in runs:
        if (b - a) * cv.rx < min_w: continue
        x0, x1 = cv.x_of_col(a), cv.x_of_col(b)
        blk = sub[:, a:b]
        # per-row "has any ink"  -> vertical extent covered
        rows = ok[blk].any(axis=1)
        cov_rows = rows.mean()
        txt = ' '.join(w[4] for w in hdr if x0 - 2 <= (w[0] + w[2]) / 2 <= x1 + 2).upper()
        hs = 0
        if 'INTERPRET' in txt: hs += 3
        if 'FINAL' in txt: hs += 2
        if 'LITHOLOG' in txt or 'LITOLOG' in txt: hs += 2
        if 'CUTTING' in txt or '%' in txt or 'DESCRIPT' in txt: hs -= 3
        cands.append(dict(x0=x0, x1=x1, cov_rows=float(cov_rows), hdr=hs, header_text=txt[:80],
                          ncolours=int(len(set(np.unique(blk)) - {0}))))
    for c in cands: c['score'] = c['hdr'] * 10 + c['cov_rows'] * 5 + min(c['ncolours'], 8) * 0.2 + (c['x1'] - c['x0']) / 200
    cands.sort(key=lambda c: -c['score'])
    return cands
