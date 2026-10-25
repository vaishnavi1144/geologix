"""Vector legend extraction: cluster drawings into swatches, pair with text labels."""
import numpy as np, collections, re
from prim import prim_hash, col

LITHO_WORDS = ('stone', 'shale', 'clay', 'sand', 'silt', 'lime', 'dolo', 'marl', 'chalk', 'coal', 'salt',
               'anhydrite', 'gypsum', 'basalt', 'trap', 'igneous', 'granite', 'metamorph', 'conglomerate',
               'breccia', 'chert', 'tuff', 'cement', 'slate', 'iron', 'grainstone', 'packer', 'wackestone',
               'mudstone', 'sample', 'volcanic', 'evaporite', 'diamictite', 'basement')

def _bbox_touch(a, b, tol=0.6):
    return not (a[2] + tol < b[0] or b[2] + tol < a[0] or a[3] + tol < b[1] or b[3] + tol < a[1])

def cluster(drs, tol=0.6):
    n = len(drs); parent = list(range(n))
    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    order = sorted(range(n), key=lambda i: drs[i]['rect'].x0)
    boxes = [tuple(d['eff']) for d in drs]
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if boxes[j][0] > boxes[i][2] + tol: break
            if _bbox_touch(boxes[i], boxes[j], tol):
                parent[f(i)] = f(j)
    out = collections.defaultdict(list)
    for i in range(n): out[f(i)].append(drs[i])
    return list(out.values())

def text_lines(words, gap=9):
    """Group words into text lines: cluster by y first, then order by x."""
    rows = []
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        for r in rows:
            if abs(r['cy'] - cy) < 3:
                r['ws'].append(w); break
        else:
            rows.append(dict(cy=cy, ws=[w]))
    lines = []
    for r in rows:
        cur = None
        for w in sorted(r['ws'], key=lambda w: w[0]):
            if cur and w[0] - cur['x1'] < gap:
                cur['t'] += ' ' + w[4]; cur['x1'] = w[2]
            else:
                cur = dict(t=w[4], x0=w[0], x1=w[2], cy=r['cy']); lines.append(cur)
    return lines

SECTION_KEYS = (('lithology', ('LITHOLOG',)), ('accessory', ('ACCESSORY',)), ('operational', ('OPERATIONAL',)))

def section_headers(lines):
    out = []
    for l in lines:
        up = l['t'].upper()
        if 'SYMBOL' in up or up.strip() in ('LITHOLOGY', 'ACCESSORY', 'OPERATIONAL'):
            for kind, keys in SECTION_KEYS:
                if any(k in up for k in keys):
                    out.append((kind, (l['x0'] + l['x1']) / 2, l['cy']))
    return out

def find_legend(page_drawings, words, header_bottom, page_width):
    """Return list of dict(name, rect, bg, prims(set), frame)"""
    dr = [d for d in page_drawings if d['eff'].y1 < header_bottom and d['eff'].width < 120 and d['eff'].height < 70]
    # swatches may be white boxes with a symbol: keep everything
    clusters = cluster(dr)
    lines = [l for l in text_lines([w for w in words if w[3] < header_bottom])]
    entries = []
    for cl in clusters:
        x0 = min(d['eff'].x0 for d in cl); y0 = min(d['eff'].y0 for d in cl)
        x1 = max(d['eff'].x1 for d in cl); y1 = max(d['eff'].y1 for d in cl)
        w, h = x1 - x0, y1 - y0
        if not (9 <= w <= 90 and 6 <= h <= 60) or len(cl) < 1: continue
        cy = (y0 + y1) / 2
        # label = nearest text line to the right on the same row
        c = [l for l in lines if l['x0'] >= x1 - 1 and l['x0'] - x1 < 45 and abs(l['cy'] - cy) < max(7, h / 2 + 3)]
        if not c: continue
        lab = min(c, key=lambda l: l['x0'] - x1)
        area = w * h
        # background = fill colour covering the most painted area inside the swatch
        # (works whether the bg is one rect or a mosaic of tiles)
        acc = collections.defaultdict(float)
        for d in cl:
            if d.get('fill') and d['type'] in ('f', 'fs'):
                acc[col(d['fill'])] += d['eff'].width * d['eff'].height
        bg = max(acc, key=acc.get) if acc and max(acc.values()) > 0.35 * area else None
        # pattern prims = stamps only: not background rect fills, not swatch frame lines
        from canvas import is_rect_fill
        def is_bg_or_frame(d):
            if d['type'] in ('f', 'fs') and is_rect_fill(d): return True
            e = d['eff']
            return d['type'] == 's' and (e.width >= 0.8 * w or e.height >= 0.8 * h)
        body = [d for d in cl if not is_bg_or_frame(d)]
        entries.append(dict(name=lab['t'].strip(), rect=(x0, y0, x1, y1), bg=bg,
                            prims=collections.Counter(prim_hash(d) for d in body), n=len(body), drawings=cl))
    heads = section_headers(lines)
    for e in entries:
        ex = (e['rect'][0] + e['rect'][2]) / 2
        above = [h for h in heads if h[2] < e['rect'][1]]
        if above:
            # nearest header above; among those prefer the closest in y then x
            yb = max(h[2] for h in above)
            row = [h for h in above if abs(h[2] - yb) < 4]
            e['section'] = min(row, key=lambda h: abs(h[1] - ex))[0]
        else:
            e['section'] = None
    return entries

JUNK = re.compile(r'(oil|gas|ohmm|units|api|ppm|fluor|mild|moderate|strong|yellow|density|neutron|lpu|g/cc|ft/hr|ms/ft|^[\\d\\W]|run|depth|total|rate|resistiv|porosity|sonic)', re.I)

def kind_of(name):
    n = name.lower()
    return 'lithology' if any(k in n for k in LITHO_WORDS) else 'other'
