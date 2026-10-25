"""Split a same-colour run into sub-intervals where the distinctive pattern stamp changes."""
import numpy as np

def split_by_stamps(seg, pool, y0, y1, win=15, dom=0.75, min_events=8):
    """seg: [(seq, drawing)] visible stamps inside the run. pool: candidate legend entries.
    Returns list of (ya, yb, name, dominance) or None if no usable evidence."""
    names = [e['name'] for e in pool]
    P = {e['name']: set(e['prims']) for e in pool}
    dist = {n: P[n] - set().union(*[P[m] for m in names if m != n]) for n in names}
    ev = []
    for _, d in seg:
        hit = [n for n in names if d['h'] in dist[n]]
        if len(hit) == 1: ev.append(((d['eff'].y0 + d['eff'].y1) / 2, hit[0]))
    if len(ev) < min_events: return None
    ev.sort(); ys = np.array([e[0] for e in ev]); lab = [e[1] for e in ev]
    # sliding-window majority label per event
    sm = []
    for i in range(len(lab)):
        a, b = max(0, i - win // 2), min(len(lab), i + win // 2 + 1)
        w = lab[a:b]; best = max(set(w), key=w.count)
        sm.append(best if w.count(best) / len(w) >= dom else (sm[-1] if sm else best))
    # hysteresis: collapse into stretches
    stretches = []; s = 0
    for i in range(1, len(sm) + 1):
        if i == len(sm) or sm[i] != sm[s]:
            stretches.append([s, i - 1, sm[s]]); s = i
    # boundaries at midpoint between last event of one stretch and first of the next
    out = []; start = y0
    for k, (a, b, n) in enumerate(stretches):
        end = y1 if k == len(stretches) - 1 else (ys[b] + ys[stretches[k + 1][0]]) / 2
        frac = sum(1 for x in lab[a:b + 1] if x == n) / (b - a + 1)
        out.append((start, end, n, round(frac, 2))); start = end
    return out
