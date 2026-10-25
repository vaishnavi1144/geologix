import os
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lx.extract import extract_page


def test_case_a_single_page_long_format():
    """Test A: A single-page, long-format PDF that currently works."""
    pdf_path = "c:/Users/vanga/Downloads/APUS_Masterlog (1).pdf"
    ref_xlsx = "c:/Users/vanga/Downloads/APUS_Masterlog (1)_geologix.xlsx"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test A file not found: {pdf_path}")

    res = extract_page(pdf_path, verbose=False)
    rows = res.get("rows", [])
    assert len(rows) > 0, "Test A produced 0 rows"

    # Check contiguity & non-overlapping
    for i in range(len(rows)):
        cur = rows[i]
        assert cur["depth_end"] > cur["depth_start"], f"Zero-length interval at {i}"
        if i > 0:
            prev = rows[i - 1]
            assert abs(cur["depth_start"] - prev["depth_end"]) < 0.001, f"Gap at {i}"

    # First / last depth within 0.2m of painted extent
    painted_top = res["painted_depth_top"]
    painted_bot = res["painted_depth_bottom"]
    assert abs(rows[0]["depth_start"] - painted_top) <= 0.2
    assert abs(rows[-1]["depth_end"] - painted_bot) <= 0.2

    # Check against reference excel match >= 98%
    if os.path.exists(ref_xlsx):
        df_ref = pd.read_excel(ref_xlsx)
        if len(df_ref) > 1 and df_ref.iloc[0].get("lithology_name") == "Unknown":
            df_ref = df_ref.iloc[1:]
        ref_bounds = sorted(set(df_ref["depth_start"].round(1).tolist() + df_ref["depth_end"].round(1).tolist()))
        bed_bounds = res.get("bed_bounds")
        if not bed_bounds:
            bed_bounds = sorted(set([r["depth_start"] for r in rows] + [r["depth_end"] for r in rows]))
        matched = sum(1 for ob in bed_bounds if any(abs(ob - rb) <= 0.25 for rb in ref_bounds))
        pct = matched / len(bed_bounds) * 100
        assert pct >= 98.0, f"Bed match ({pct:.1f}%) is below 98%"


def test_case_b_two_page_lithology():
    """Test B: A two-page lithology PDF (both pages processed and merged)."""
    pdf_path = "c:/Users/vanga/Downloads/ONGC Onshore Masterlog.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test B file not found: {pdf_path}")

    res = extract_page(pdf_path, verbose=False)
    rows = res.get("rows", [])
    assert len(rows) > 0, "Test B produced 0 rows"

    # Verify both pages were processed and reached page 2 depth extent (> 3000m)
    # Page 1 alone stopped at ~2883m; the full PDF extends to 3215.91m
    assert rows[0]["depth_start"] <= 503.0, f"Starting depth {rows[0]['depth_start']} unexpected"
    assert rows[-1]["depth_end"] >= 3200.0, (
        f"Extraction stopped before page 2! Final depth: {rows[-1]['depth_end']}"
    )

    # Verify contiguity and non-overlapping across the entire log
    for i in range(len(rows)):
        cur = rows[i]
        assert cur["depth_end"] > cur["depth_start"]
        if i > 0:
            prev = rows[i - 1]
            assert abs(cur["depth_start"] - prev["depth_end"]) < 0.001

    # First and last depth match total painted extent
    assert abs(rows[0]["depth_start"] - res["painted_depth_top"]) <= 0.2
    assert abs(rows[-1]["depth_end"] - res["painted_depth_bottom"]) <= 0.2


def test_case_c_three_or_more_pages():
    """Test C: A PDF containing three or more pages."""
    pdf_path = PROJECT_ROOT / "tests" / "test_c_3pages.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test C file not found: {pdf_path}")

    res = extract_page(str(pdf_path), verbose=False)
    rows = res.get("rows", [])
    assert len(rows) > 0, "Test C produced 0 rows"

    # Verify all 3 pages were processed
    assert rows[0]["depth_start"] <= 503.0
    assert rows[-1]["depth_end"] >= 3200.0

    # Strict contiguity
    for i in range(len(rows)):
        cur = rows[i]
        assert cur["depth_end"] > cur["depth_start"]
        if i > 0:
            prev = rows[i - 1]
            assert abs(cur["depth_start"] - prev["depth_end"]) < 0.001


def test_case_d_boundary_lithology_continuity():
    """Test D: Multi-page PDF where a depth interval or lithology pattern continues across a page boundary."""
    pdf_path = "c:/Users/vanga/Downloads/ONGC Onshore Masterlog.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test D file not found: {pdf_path}")

    res = extract_page(pdf_path, verbose=False)
    rows = res.get("rows", [])

    # In ONGC Onshore Masterlog, Page 0 bottom is Shale (ending at 2883.06m)
    # and Page 1 top is Shale (starting at 2890.39m).
    # Consolidation must merge them into a single continuous Shale bed spanning the boundary.
    boundary_rows = [r for r in rows if r["depth_start"] <= 2883.06 and r["depth_end"] >= 2890.39]
    assert len(boundary_rows) == 1, (
        f"Expected exactly 1 merged boundary row spanning 2883-2890m, found {len(boundary_rows)}: {boundary_rows}"
    )
    assert boundary_rows[0]["lithology"] == "Shale", (
        f"Expected boundary lithology 'Shale', got {boundary_rows[0]['lithology']}"
    )
    assert boundary_rows[0]["depth_start"] <= 2860.0
    assert boundary_rows[0]["depth_end"] >= 2895.0


def test_case_e_repeated_headers_and_overlapping_depths():
    """Test E: Multi-page PDF containing repeated headers, overlapping depth labels, or page margins."""
    pdf_path = PROJECT_ROOT / "tests" / "test_e_overlap.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test E file not found: {pdf_path}")

    res = extract_page(str(pdf_path), verbose=False)
    rows = res.get("rows", [])
    assert len(rows) > 0, "Test E produced 0 rows"

    # Verify no duplicate or reversed intervals were introduced by the overlap
    for i in range(len(rows)):
        cur = rows[i]
        assert cur["depth_end"] > cur["depth_start"]
        if i > 0:
            prev = rows[i - 1]
            assert abs(cur["depth_start"] - prev["depth_end"]) < 0.001, (
                f"Discontinuity at row {i}: {prev['depth_end']} != {cur['depth_start']}"
            )
            assert cur["depth_start"] >= prev["depth_start"], "Depths went backwards!"
