# GeologiX

Automated Well Log Analysis & Lithology Extractor from borehole PDF logs.

## Overview

GeologiX is an automated tool designed to parse geological borehole PDFs, detect lithological columns, analyze stratigraphic patterns/depth markers, and export structured interval records to Excel spreadsheets (`.xlsx`).

## Features

- **Automated Column Localization**: Detects lithology track boundaries and scales across PDF pages.
- **Dual Raster & Vector Extraction**: Analyzes vector geometries and scanned raster logs.
- **Depth Calibration**: Reconstructs depth scales with configurable units (`m` / `ft`).
- **Interactive Web Interface**: Modern UI to upload borehole PDFs, inspect extracted intervals, review confidence scores, and download structured Excel files.
- **REST API**: Built with Flask for programmatic extraction and Excel generation.

## Project Structure

```text
├── backend/
│   └── app.py              # Flask server and REST API endpoints
├── frontend/
│   ├── index.html          # Web application UI
│   ├── app.js              # Frontend logic & API interaction
│   └── styles.css          # Design system styling
├── lx/
│   ├── extract.py          # High-level extraction pipeline
│   ├── locate.py           # Column boundary detection
│   ├── raster.py           # Raster pattern analysis
│   ├── canvas.py           # Page rendering and canvas geometry
│   ├── column.py           # Lithological interval parsing
│   └── depthcal.py         # Depth axis calibration
├── tests/                  # Test suites & sample verification PDFs
├── validate_vs_vectors.py  # Vector verification script
└── requirements.txt        # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python backend/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 3. Run Verification & Tests

```bash
python -m pytest
python validate_vs_vectors.py tests/test_c_3pages.pdf
```
