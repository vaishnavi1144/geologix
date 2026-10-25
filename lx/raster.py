import numpy as np
import pymupdf as fitz
from depthcal import calibrate, depth_at
import re

# Standard geological lithology color references in RGB
STD_LITHOLOGY_COLORS = [
    ("Sandstone", np.array([245, 230, 80], dtype=np.float32)),        # Yellow
    ("Claystone", np.array([140, 150, 130], dtype=np.float32)),       # Olive / Grey-green
    ("Shale", np.array([120, 120, 120], dtype=np.float32)),           # Medium grey
    ("Limestone", np.array([100, 180, 240], dtype=np.float32)),       # Light blue
    ("Dolomite", np.array([210, 140, 210], dtype=np.float32)),        # Purple / Magenta
    ("Siltstone", np.array([195, 165, 100], dtype=np.float32)),       # Tan / Ochre-brown
    ("Coal", np.array([45, 45, 45], dtype=np.float32)),                # Black / Dark
    ("Anhydrite", np.array([215, 210, 235], dtype=np.float32)),       # Pale violet
    ("Marl", np.array([160, 200, 175], dtype=np.float32)),            # Pale greenish grey
    ("Halite", np.array([245, 180, 160], dtype=np.float32)),          # Pale orange / Salmon
    ("Volcanics", np.array([200, 80, 80], dtype=np.float32)),         # Dull red
    ("Chalk", np.array([235, 235, 230], dtype=np.float32)),           # Chalk white/grey
]

def classify_color(rgb):
    """Finds the closest standard lithology for a given RGB array."""
    c = np.array(rgb, dtype=np.float32)
    # Check if coal / very dark
    if float(np.mean(c)) < 55:
        return "Coal", 0.90
    
    best_name = "Claystone"
    best_dist = float('inf')
    for name, ref in STD_LITHOLOGY_COLORS:
        dist = float(np.linalg.norm(c - ref))
        if dist < best_dist:
            best_dist = dist
            best_name = name
            
    conf = float(max(0.50, min(0.95, 1.0 - (best_dist / 300.0))))
    return best_name, float(round(conf, 2))

def extract_raster_page(doc, page_no=None, verbose=False):
    """
    Tier 2 Computer Vision Fallback:
    Extracts lithology intervals from scanned or raster-based well logs.
    """
    if page_no is not None:
        pages_to_check = [page_no]
    else:
        # Check all pages, prioritising pages with highest vertical extent
        pages_to_check = sorted(range(len(doc)), key=lambda i: doc[i].rect.height, reverse=True)

    last_error = "No valid raster lithology found."
    for p_idx in pages_to_check:
        p = doc[p_idx]
        try:
            res = _process_single_raster_page(p, verbose=verbose)
            if res and len(res.get("rows", [])) > 0:
                res["page"] = int(p_idx)
                return res
        except Exception as exc:
            last_error = str(exc)
            continue

    raise ValueError(f"Raster/Scanned document extraction failed: {last_error}")

def _process_single_raster_page(p, verbose=False):
    # Render at 150 DPI for balance of fine resolution and fast processing
    dpi = 150
    pix = p.get_pixmap(dpi=dpi)
    w, h = pix.width, pix.height
    n = pix.n
    
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, n))
    if n >= 3:
        img = img[:, :, :3]
    else:
        raise ValueError("Image is monochrome; no colour channels present.")

    # 1. Depth Calibration via OCR/Text layer if available
    words = p.get_text('words')
    cal = None
    depth_unit = 'm'
    scale_y = float(p.rect.height / float(h))
    
    if len(words) >= 5:
        try:
            cal = calibrate(words)
            step = cal.get('grid', abs(cal.get('last', 0) - cal.get('first', 0)) / max(1, cal.get('n', 1) - 1))
            max_d = max(cal.get('first', 0), cal.get('last', 0))
            is_step_5_10 = (abs(step - 5) < 0.6 or abs(step - 10) < 0.6)
            has_ft_label = any(
                w[4].strip().lower() in ('ft', '(ft)', 'feet', '(feet)', 'depth(ft)', 'depth(feet)')
                for w in words
                if abs((w[0] + w[2]) / 2 - (cal['x'][0] + cal['x'][1]) / 2) < 60 and w[1] < cal['top'] + 50
            )
            is_feet = bool((max_d > 5000 and is_step_5_10) or (has_ft_label and max_d > 1000) or (has_ft_label and step >= 50))
            depth_unit = 'ft' if is_feet else 'm'
        except Exception:
            cal = None

    # 2. Identify the coloured lithology column using color saturation profiling
    R = img[:, :, 0].astype(np.int16)
    G = img[:, :, 1].astype(np.int16)
    B = img[:, :, 2].astype(np.int16)
    
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    sat = max_c - min_c
    bright = (max_c + min_c) // 2

    # Colored lithology pixels: distinct saturation, not pure background white or grid black
    col_mask = (sat > 25) & (bright > 30) & (bright < 242)
    
    # Calculate vertical profile across column x
    col_density = np.sum(col_mask, axis=0).astype(np.float32)
    
    # Smooth horizontal profile with window
    win = max(5, int(w * 0.015))
    kernel = np.ones(win) / win
    smoothed = np.convolve(col_density, kernel, mode='same')
    
    # Threshold for candidate columns
    min_coverage = 0.08 * h
    active_x = np.where(smoothed > min_coverage)[0]
    
    if len(active_x) == 0:
        raise ValueError("No coloured lithology column found in scanned document.")
        
    # Find contiguous spans of active x
    spans = []
    start = int(active_x[0])
    prev = start
    for x in active_x[1:]:
        x_int = int(x)
        if x_int == prev + 1:
            prev = x_int
        else:
            if (prev - start) >= int(w * 0.015):
                spans.append((start, prev))
            start = x_int
            prev = x_int
    if (prev - start) >= int(w * 0.015):
        spans.append((start, prev))
        
    if not spans:
        raise ValueError("No candidate lithology column of sufficient width found.")
        
    # Score spans by total color mass within valid column width range (1.5% to 25% of page width)
    best_span = None
    best_score = -1.0
    for s0, s1 in spans:
        span_w = s1 - s0
        if span_w < int(w * 0.015) or span_w > int(w * 0.35):
            continue
        score = float(np.sum(smoothed[s0:s1+1]))
        if score > best_score:
            best_score = score
            best_span = (s0, s1)
            
    if best_span is None:
        best_span = spans[0]
        
    cx0, cx1 = best_span
    
    # 3. Define vertical extent of colored column
    inner_margin = max(2, int((cx1 - cx0) * 0.15))
    ix0 = cx0 + inner_margin
    ix1 = cx1 - inner_margin
    if ix1 <= ix0:
        ix0, ix1 = cx0, cx1
        
    column_strip = img[:, ix0:ix1, :]
    col_sat = sat[:, ix0:ix1]
    col_bright = bright[:, ix0:ix1]
    row_colored = np.sum((col_sat > 25) & (col_bright > 30) & (col_bright < 242), axis=1)
    
    active_y = np.where(row_colored > 0)[0]
    if len(active_y) < 10:
        raise ValueError("Coloured column does not contain sufficient vertical intervals.")
        
    y_start_px = int(active_y[0])
    y_end_px = int(active_y[-1])
    
    # 4. Horizontal bed segmentation down column_strip
    row_colors = np.median(column_strip[y_start_px:y_end_px + 1], axis=1)
    
    # Merge consecutive rows into intervals where color distance < threshold
    intervals_px = []
    cur_y0 = y_start_px
    cur_color_sum = row_colors[0].astype(np.float32)
    cur_count = 1
    
    for idx in range(1, len(row_colors)):
        y_curr = y_start_px + idx
        c_curr = row_colors[idx]
        mean_c = cur_color_sum / cur_count
        diff = float(np.linalg.norm(c_curr - mean_c))
        
        # If color difference exceeds threshold or hit max bed size
        if diff > 24.0 and cur_count >= 3:
            intervals_px.append((cur_y0, y_curr, mean_c))
            cur_y0 = y_curr
            cur_color_sum = c_curr.astype(np.float32)
            cur_count = 1
        else:
            cur_color_sum += c_curr
            cur_count += 1
            
    if cur_count > 0:
        intervals_px.append((cur_y0, y_end_px, cur_color_sum / cur_count))
        
    # 5. Map pixel intervals to depth
    rows = []
    if cal:
        for y0_p, y1_p, c_avg in intervals_px:
            pt0 = y0_p * scale_y
            pt1 = y1_p * scale_y
            d0 = float(depth_at(cal, pt0))
            d1 = float(depth_at(cal, pt1))
            if d0 > d1:
                d0, d1 = d1, d0
            lith_name, conf = classify_color(c_avg)
            rows.append({
                "depth_start": float(round(d0, 2)),
                "depth_end": float(round(d1, 2)),
                "lithology": str(lith_name),
                "confidence": float(round(conf, 2)),
                "method": "raster_vision",
                "runner_up": "",
            })
    else:
        # Nominal depth scaling when text axis is absent
        nom_top = 1000.0
        nom_scale = 0.5  # 0.5m per pixel row
        for y0_p, y1_p, c_avg in intervals_px:
            d0 = float(nom_top + (y0_p - y_start_px) * nom_scale)
            d1 = float(nom_top + (y1_p - y_start_px) * nom_scale)
            lith_name, conf = classify_color(c_avg)
            rows.append({
                "depth_start": float(round(d0, 2)),
                "depth_end": float(round(d1, 2)),
                "lithology": str(lith_name),
                "confidence": float(round(conf, 2)),
                "method": "raster_vision (nominal scale)",
                "runner_up": "",
            })

    # Filter out 0-thickness intervals
    rows = [r for r in rows if r["depth_end"] > r["depth_start"]]

    return {
        "rows": rows,
        "unit": depth_unit,
        "marker_mode": "raster_vision",
        "markers": [],
        "warnings": ["Processed via Computer Vision Raster Fallback (scanned document)."],
        "column": [float(cx0 * (p.rect.width / float(w))), float(cx1 * (p.rect.width / float(w)))],
    }
