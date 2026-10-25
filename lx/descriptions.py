"""Description-text anchors: 'LITHOLOGY: description...' blocks positioned at depth."""
import re, collections
from legend import text_lines, LITHO_WORDS

HEAD = re.compile(r'^(?:Segments?\s*[\d\-–, ]+:\s*)?(?:Trace\s+|Rare\s+)?([A-Za-z][A-Za-z/ \-]{2,40}?)\s*:\s')

def norm(n):
    return re.sub(r'[^a-z ]', '', n.lower()).strip()

def description_anchors(words, y_min, y_max, gap=6):
    """Return list of dict(y, names[list], text) for description headers inside the chart y-range."""
    ws = [w for w in words if y_min <= (w[1] + w[3]) / 2 <= y_max and w[2] - w[0] < 400]
    lines = text_lines(ws, gap=gap)
    out = []
    for l in lines:
        m = HEAD.match(l['t'])
        if not m: continue
        raw = m.group(1)
        names = [norm(x) for x in re.split(r'[/,]| and ', raw)]
        names = [n for n in names if any(k in n for k in LITHO_WORDS)]
        if names: out.append(dict(y=l['cy'], names=names, raw=raw.strip(), x=l['x0']))
    return out

def agree(interval_name, anchor_names):
    """Loose token agreement: 'Silty Shale' ~ 'shale'; 'Calcareous Claystone' ~ 'claystone'."""
    if not interval_name: return None
    toks = set(norm(interval_name).split())
    base = norm(interval_name).split()[-1] if norm(interval_name) else ''
    for a in anchor_names:
        at = set(a.split())
        if norm(interval_name) == a or base in at or a.split()[-1] in toks: return True
    return False
