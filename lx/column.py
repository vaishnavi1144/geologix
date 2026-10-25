"""Lithology column: locate (x-range), build a z-order-correct visibility map, segment, classify."""
import numpy as np, collections, re
from prim import col

WHITE = (1.0, 1.0, 1.0)

def is_white(c): return c is None or all(v > 0.97 for v in c)

# ----------------------------------------------------------------------------- locate
def column_candidates(drs, cal, words, min_w=8):
    """Candidate lithology-column x-extents = x-extents of large opaque fills, ranked."""
    top, bot = cal['top'] - 20, cal['bottom'] + 20
    ext = collections.defaultdict(lambda: [0.0, set()])
    for d in drs:
        e = d['eff']
        if d['type'] not in ('f', 'fs') or not d.get('fill') or is_white(d['fill']): continue
        if e.width < min_w or e.y0 < top or e.y1 > bot or e.height < 1: continue
        k = (round(e.x0 * 2) / 2, round(e.x1 * 2) / 2)
        ext[k][0] += e.height; ext[k][1].add(col(d['fill']))
    # merge near-identical extents
    cands = []
    for (x0, x1), (h, cols) in ext.items():
        if x1 - x0 < min_w: continue
        cands.append(dict(x0=x0, x1=x1, height=h, colours=cols))
    merged = []
    for c in sorted(cands, key=lambda c: -c['height']):
        for m in merged:
            if abs(m['x0'] - c['x0']) < 3 and abs(m['x1'] - c['x1']) < 3:
                m['height'] += c['height']; m['colours'] |= c['colours']; break
        else: merged.append(dict(c))
    # header text above chart over each extent
    hdr = [w for w in words if cal['top'] - 420 < w[1] < cal['top'] + 40]
    for m in merged:
        txt = ' '.join(w[4] for w in hdr if w[2] > m['x0'] - 4 and w[0] < m['x1'] + 4).upper()
        s = 0
        if 'INTERPRET' in txt: s += 3
        if 'FINAL' in txt: s += 3
        if 'LITHOLOG' in txt or 'LITOLOG' in txt: s += 2
        if 'CUTTING' in txt or '%' in txt: s -= 3
        m['hdr_score'] = s
    for m in merged:
        m['score'] = m['hdr_score'] * 1e5 + m['height'] * (1 + 0.1 * len(m['colours']))
    merged.sort(key=lambda m: -m['score'])
    return merged

# ------------------------------------------------------------------- visibility map
def visibility(drs, cal, x0, x1, res_pt=0.5):
    """Painter's algorithm along y inside [x0,x1].
    Returns rows (n), y_top, arrays: paint_seq[n], paint_col[n] (colour tuple/None) and visible prim events."""
    top, bot = cal['top'] - 30, cal['bottom'] + 30
    n = int((bot - top) / res_pt) + 1
    paint_seq = np.full(n, -1, dtype=np.int64)
    paint_col = [None] * n
    cw = x1 - x0
    prims = []   # (seq, r0, r1, hash)
    seq = 0
    for d in drs:
        e = d['eff']; seq += 1
        if e.y1 < top or e.y0 > bot or e.x1 < x0 + 1 or e.x0 > x1 - 1: continue
        r0 = max(0, int((e.y0 - top) / res_pt)); r1 = min(n, int(np.ceil((e.y1 - top) / res_pt)) + 1)
        ow = min(e.x1, x1) - max(e.x0, x0)
        if d['type'] in ('f', 'fs') and d.get('fill') and ow >= 0.9 * cw and e.height > 0.8:
            paint_seq[r0:r1] = seq
            c = col(d['fill'])
            for r in range(r0, r1): paint_col[r] = c
        else:
            if e.x0 >= x0 - 1 and e.x1 <= x1 + 1 and (e.width < 0.9 * cw or d['type'] == 's'):
                prims.append((seq, r0, r1, d['h'], d))
    vis = [(s, a, b, h, dd) for s, a, b, h, dd in prims if s > paint_seq[min(a, n - 1)]]
    return dict(top=top, res=res_pt, n=n, paint_col=paint_col, paint_seq=paint_seq, prims=vis)

# ------------------------------------------------------ vector descriptor (fallback)
def vec_descriptor(dlist, area):
    """Scale-tolerant descriptor of a set of pattern drawings (stroke angles, sizes, ink)."""
    ang = np.zeros(8); size = np.zeros(6); kinds = collections.Counter()
    for d in dlist:
        kinds[d['type']] += 1
        for it in d['items']:
            if it[0] == 'l':
                p, q = it[1], it[2]
                dx, dy = q.x - p.x, q.y - p.y
                L = (dx * dx + dy * dy) ** .5
                if L < 1e-3: continue
                a = (np.degrees(np.arctan2(dy, dx)) % 180) / 22.5
                ang[int(a) % 8] += L
            elif it[0] == 'c': ang[0] += 0.3
        r = d['eff']; s = max(r.width, r.height)
        size[min(5, int(s / 1.5))] += 1
    ang = ang / (ang.sum() + 1e-9); size = size / (size.sum() + 1e-9)
    dens = sum(d['eff'].width * d['eff'].height for d in dlist) / max(area, 1e-6)
    return np.concatenate([ang, size, [min(dens, 1.0), kinds['f'] / (sum(kinds.values()) + 1e-9)]])
