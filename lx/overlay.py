import sys; sys.path.insert(0,'/home/claude/lx')
import pymupdf as fitz
from PIL import Image, ImageDraw
import io

def overlay(res, out, y_from=None, y_to=None, dpi=60, pad=6):
    p = res['page']; xa, xb = res['column']; cal = res['cal']
    y_from = y_from if y_from is not None else cal['top'] - 10
    y_to = y_to if y_to is not None else cal['bottom'] + 10
    clip = fitz.Rect(max(0, xa - 70), y_from, min(p.rect.width, xb + 15), y_to)
    pm = p.get_pixmap(clip=clip, dpi=dpi)
    im = Image.open(io.BytesIO(pm.tobytes('png'))).convert('RGB')
    s = dpi / 72.0
    W = im.width + 240
    canvas = Image.new('RGB', (W, im.height), 'white'); canvas.paste(im, (0, 0))
    dr = ImageDraw.Draw(canvas)
    for r in res['rows']:
        for y in (r['y0'], r['y1']):
            yy = (y - y_from) * s
            if 0 <= yy <= im.height: dr.line([(0, yy), (im.width + 4, yy)], fill=(255, 0, 0), width=1)
        ym = ((r['y0'] + r['y1']) / 2 - y_from) * s
        if 0 <= ym <= im.height and r['lithology']:
            dr.text((im.width + 8, ym - 5), f"{r['lithology'][:22]} {r['depth_start']:.0f}-{r['depth_end']:.0f}", fill=(0, 0, 160))
    canvas.save(out)
