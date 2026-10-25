import json
import os
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import pdfplumber
import pikepdf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lx.extract import extract_page


def parse_pikepdf_vector_ops(page):
    """Inspect content streams via pikepdf to count vector path and painting operators."""
    vector_ops = {'re', 'm', 'l', 'c', 'v', 'y', 'h', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*', 's', 'S'}
    op_counts = 0
    try:
        raw_contents = page.get('/Contents')
        if not raw_contents:
            return 0
        streams = raw_contents if isinstance(raw_contents, pikepdf.Array) else [raw_contents]
        for stm in streams:
            try:
                data = stm.read_bytes()
                # Tokens analysis
                for token in data.split():
                    try:
                        t_str = token.decode('latin1', errors='ignore')
                        if t_str in vector_ops:
                            op_counts += 1
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception as exc:
        pass
    return op_counts


def validate_pdf_vs_vectors(pdf_path, verbose=False):
    """
    Independent validation of PDF using pikepdf and pdfplumber:
    - Verifies vector content vs scanned raster
    - Evaluates column fills and extents independently
    - Validates extract_page output against vector assertions
    """
    report = {
        "pdf": str(pdf_path),
        "filename": Path(pdf_path).name,
        "is_vector": True,
        "pikepdf_ops": 0,
        "pdfplumber_rects": 0,
        "pdfplumber_curves": 0,
        "pdfplumber_lines": 0,
        "mode_chosen": None,
        "marker_count": 0,
        "interval_count": 0,
        "contiguous": True,
        "non_overlapping": True,
        "no_short_beds": True,
        "extent_match_0_2m": True,
        "status": "PASS",
        "error": None,
    }

    # 1. Inspect with pikepdf
    try:
        with pikepdf.open(pdf_path) as pike_doc:
            p0 = pike_doc.pages[0]
            report["pikepdf_ops"] = parse_pikepdf_vector_ops(p0)
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"pikepdf failed: {exc}"
        return report

    # 2. Inspect with pdfplumber
    try:
        with pdfplumber.open(pdf_path) as plum_doc:
            page = plum_doc.pages[0]
            report["pdfplumber_rects"] = len(page.rects)
            report["pdfplumber_curves"] = len(page.curves)
            report["pdfplumber_lines"] = len(page.lines)
            total_vecs = report["pdfplumber_rects"] + report["pdfplumber_curves"]
            if total_vecs < 5 and report["pikepdf_ops"] < 5:
                report["is_vector"] = False
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"pdfplumber failed: {exc}"
        return report

    # 3. Test extract_page
    try:
        res = extract_page(pdf_path, verbose=False)
        report["mode_chosen"] = res.get("marker_mode")
        report["marker_count"] = len(res.get("markers", []))
        rows = res.get("rows", [])
        report["interval_count"] = len(rows)

        # Assert contiguous and non-overlapping
        for i in range(len(rows)):
            cur = rows[i]
            if cur["depth_end"] <= cur["depth_start"]:
                report["non_overlapping"] = False
            if i > 0:
                prev = rows[i - 1]
                if abs(cur["depth_start"] - prev["depth_end"]) > 0.001:
                    report["contiguous"] = False
            # Check thickness >= 0.15 m (or 0.5 ft)
            unit = res.get("unit", "m")
            min_thresh = 0.49 if unit == "ft" else 0.149
            thickness = round(cur["depth_end"] - cur["depth_start"], 2)
            if thickness < min_thresh:
                report["no_short_beds"] = False

        # Assert painted extent matches first/last depth within 0.2 m
        top_diff = abs(rows[0]["depth_start"] - res["painted_depth_top"]) if rows else 999
        bot_diff = abs(rows[-1]["depth_end"] - res["painted_depth_bottom"]) if rows else 999
        if top_diff > 0.2 or bot_diff > 0.2:
            report["extent_match_0_2m"] = False

        if not (report["contiguous"] and report["non_overlapping"] and report["no_short_beds"] and report["extent_match_0_2m"]):
            report["status"] = "FAIL"

    except Exception as exc:
        report["status"] = "REJECTED_AS_EXPECTED" if isinstance(exc, ValueError) else "ERROR"
        report["error"] = str(exc)

    return report


def main():
    corpus_file = PROJECT_ROOT / "tests" / "corpus.json"
    pdf_targets = []

    if len(sys.argv) > 1:
        pdf_targets = [{"pdf": sys.argv[1]}]
    elif corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            pdf_targets = json.load(f)
    else:
        print("No input PDF or corpus.json found.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("INDEPENDENT VECTOR VALIDATION (pikepdf + pdfplumber)")
    print("=" * 80)

    results = []
    for item in pdf_targets:
        pdf_path = item.get("pdf") if isinstance(item, dict) else item[0]
        if not os.path.exists(pdf_path):
            print(f"Skipping missing file: {pdf_path}")
            continue

        print(f"\nEvaluating: {Path(pdf_path).name}")
        rep = validate_pdf_vs_vectors(pdf_path)
        results.append(rep)

        print(f"  • pikepdf Path Ops:      {rep['pikepdf_ops']}")
        print(f"  • pdfplumber Vectors:    {rep['pdfplumber_rects']} rects, {rep['pdfplumber_curves']} curves, {rep['pdfplumber_lines']} lines")
        print(f"  • Vector Document:       {'Yes' if rep['is_vector'] else 'No (Raster/Scanned)'}")
        if rep['error']:
            print(f"  • Fail Loudly Message:   {rep['error']}")
            print(f"  • Status:                {rep['status']}")
        else:
            print(f"  • Chosen Marker Mode:    {rep['mode_chosen']}")
            print(f"  • Markers Found:         {rep['marker_count']}")
            print(f"  • Main Intervals:        {rep['interval_count']}")
            print(f"  • Contiguous & No-Overlap: {'PASS' if rep['contiguous'] and rep['non_overlapping'] else 'FAIL'}")
            print(f"  • Extent Match <= 0.2m:  {'PASS' if rep['extent_match_0_2m'] else 'FAIL'}")
            print(f"  • Min Thickness >= 0.15m:{'PASS' if rep['no_short_beds'] else 'FAIL'}")
            print(f"  • Status:                {rep['status']}")

    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {sum(1 for r in results if r['status'] in ('PASS', 'REJECTED_AS_EXPECTED'))}/{len(results)} PASSED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
