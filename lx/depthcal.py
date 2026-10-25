"""Depth axis calibration from numeric labels (vector text layer)."""
import re, numpy as np, collections

NUM = re.compile(r'^-?\d{1,5}(\.\d+)?$')

def _fit(pts):
    y = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts])
    a, b = np.polyfit(y, d, 1)
    res = d - (a * y + b)
    ss = ((d - d.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / ss if ss > 0 else 0
    return a, b, r2, res

def calibrate(words, min_labels=5, min_r2=0.998):
    """words: pymupdf words. Returns dict(a,b,r2,n,x,top,bottom) mapping y_pt -> depth = a*y+b."""
    cands = [w for w in words if NUM.match(w[4])]
    groups = collections.defaultdict(list)
    for w in cands:
        groups[round((w[0] + w[2]) / 2 / 4)].append(w)      # bucket by x-centre (4pt)
    best = None
    for k, ws in groups.items():
        for kk in (k, k + 1):                               # merge neighbouring buckets
            ws2 = groups.get(k, []) + groups.get(k + 1, [])
        pts = sorted(((w[1] + w[3]) / 2, float(w[4])) for w in ws2)
        pts = list(dict.fromkeys(pts))
        while len(pts) >= min_labels:
            a, b, r2, res = _fit(pts)
            if r2 >= min_r2 and abs(a) > 1e-4:
                break
            pts.pop(int(np.argmax(np.abs(res))))            # drop worst outlier
        else:
            continue
        # unit sanity: monotonic
        d = [p[1] for p in pts]
        if not (all(x < y for x, y in zip(d, d[1:])) or all(x > y for x, y in zip(d, d[1:]))):
            continue
        diffs = np.diff(d)
        grid = float(np.median(diffs)) if len(diffs) > 0 else 0
        cand = dict(a=a, b=b, r2=r2, n=len(pts), x=(min(w[0] for w in ws2), max(w[2] for w in ws2)),
                    top=pts[0][0], bottom=pts[-1][0], first=pts[0][1], last=pts[-1][1], grid=grid)

        # Prioritize standard round grid steps (e.g. 1, 2, 5, 10, 20, 25, 50, 100, 250)
        # and integer depth values over raw count alone (to avoid pipe tally tracks)
        is_round_step = False
        is_standard_grid = False
        if grid > 0:
            r_step = round(grid)
            if abs(grid - r_step) < 0.05:
                is_round_step = True
                if r_step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000) or r_step % 5 == 0:
                    is_standard_grid = True
        int_frac = sum(1 for v in d if abs(v - round(v)) < 0.01) / max(1, len(d))
        cand['score'] = (1000.0 if is_standard_grid else (500.0 if is_round_step else 0.0)) + int_frac * 200.0 + cand['n']

        if best is None or cand['score'] > best['score']:
            best = cand
    if best is None:
        raise RuntimeError('no depth axis found')
    # invert: a is depth per pt; pt_per_unit
    best['pt_per_unit'] = 1.0 / best['a']
    labs = sorted(set(float(w[4]) for w in cands if abs(((w[1]+w[3])/2) * best['a'] + best['b'] - float(w[4])) < 0.5))
    if len(labs) > 2:
        best['grid'] = float(np.median(np.diff(labs)))
    return best

def depth_at(cal, y): return cal['a'] * y + cal['b']
def y_at(cal, d): return (d - cal['b']) / cal['a']
