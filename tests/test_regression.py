import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lx.extract import extract_page
from validate_vs_vectors import validate_pdf_vs_vectors

CORPUS_PATH = PROJECT_ROOT / "tests" / "corpus.json"


def load_corpus():
    if not CORPUS_PATH.exists():
        return []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for entry in data:
        if isinstance(entry, dict):
            items.append((entry.get("pdf"), entry.get("reference_xlsx"), entry.get("expected_error")))
        elif isinstance(entry, (list, tuple)):
            ref = entry[1] if len(entry) > 1 else None
            exp = entry[2] if len(entry) > 2 else None
            items.append((entry[0], ref, exp))
    return items


@pytest.mark.parametrize("pdf_path,ref_xlsx,expected_error", load_corpus())
def test_pdf_extraction(pdf_path, ref_xlsx, expected_error):
    if not os.path.exists(pdf_path):
        pytest.skip(f"PDF file does not exist: {pdf_path}")

    # Case 1: Expected fail-loudly error
    if expected_error:
        with pytest.raises(ValueError) as exc_info:
            extract_page(pdf_path)
        err_msg = str(exc_info.value).lower()
        assert any(k in err_msg for k in ["legend", "feet", "vector", "column", "depth axis"]), (
            f"Error message did not match expected fail-loudly pattern: {exc_info.value}"
        )
        return

    # Case 2: Valid PDF extraction
    res = extract_page(pdf_path, verbose=False)
    rows = res.get("rows", [])
    markers = res.get("markers", [])
    assert len(rows) > 0, "Extraction produced 0 rows"

    # 1. Contiguous and non-overlapping
    for i in range(len(rows)):
        cur = rows[i]
        assert cur["depth_end"] > cur["depth_start"], (
            f"Interval {i} is overlapping or zero-length: {cur['depth_start']} -> {cur['depth_end']}"
        )
        if i > 0:
            prev = rows[i - 1]
            assert abs(cur["depth_start"] - prev["depth_end"]) < 0.001, (
                f"Intervals {i-1} and {i} are not contiguous: {prev['depth_end']} != {cur['depth_start']}"
            )

    # 2. First/last depth match painted column extent within 0.2 m
    painted_top = res["painted_depth_top"]
    painted_bot = res["painted_depth_bottom"]
    first_start = rows[0]["depth_start"]
    last_end = rows[-1]["depth_end"]

    assert abs(first_start - painted_top) <= 0.2, (
        f"First depth {first_start} does not match painted top {painted_top} within 0.2 m (diff: {abs(first_start - painted_top):.2f})"
    )
    assert abs(last_end - painted_bot) <= 0.2, (
        f"Last depth {last_end} does not match painted bottom {painted_bot} within 0.2 m (diff: {abs(last_end - painted_bot):.2f})"
    )

    # 3. No interval shorter than 0.15 m (or 0.5 ft) unless marked
    unit = res.get("unit", "m")
    min_thresh = 0.49 if unit == "ft" else 0.149
    for i, cur in enumerate(rows):
        th = round(cur["depth_end"] - cur["depth_start"], 2)
        assert th >= min_thresh, f"Row {i} interval is shorter than {min_thresh} {unit} ({th} {unit}): {cur}"

    # 4. Where a reference xlsx exists, bed-level match >= 98% within 0.25 m
    if ref_xlsx and os.path.exists(ref_xlsx):
        df_ref = pd.read_excel(ref_xlsx)
        # Drop initial unpainted gap row if present
        if len(df_ref) > 1 and df_ref.iloc[0].get("lithology_name") == "Unknown":
            df_ref = df_ref.iloc[1:]

        ref_bounds = sorted(set(df_ref["depth_start"].round(1).tolist() + df_ref["depth_end"].round(1).tolist()))
        bed_bounds = res.get("bed_bounds")
        if not bed_bounds:
            bed_bounds = sorted(set([r["depth_start"] for r in rows] + [r["depth_end"] for r in rows]))

        matched = sum(1 for ob in bed_bounds if any(abs(ob - rb) <= 0.25 for rb in ref_bounds))
        pct = matched / len(bed_bounds) * 100
        assert pct >= 98.0, f"Bed-level match against reference ({pct:.1f}%) is below 98% within 0.25m."


def run_all_harness():
    corpus = load_corpus()
    print("\n" + "=" * 80)
    print("REGRESSION TEST HARNESS")
    print("=" * 80)

    passed = 0
    failed = 0

    for item in corpus:
        pdf_path, ref_xlsx, expected_error = item
        name = Path(pdf_path).name
        print(f"\n[TESTING] {name}")
        try:
            test_pdf_extraction(pdf_path, ref_xlsx, expected_error)
            print(f"  --> RESULT: PASSED")
            passed += 1
        except Exception as e:
            print(f"  --> RESULT: FAILED ({e})")
            failed += 1

    print("\n" + "=" * 80)
    print(f"REGRESSION SUITE COMPLETED: {passed} PASSED, {failed} FAILED")
    print("=" * 80 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_harness()
    sys.exit(0 if success else 1)
