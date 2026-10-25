const fileInput = document.getElementById('pdfFile');
const filePromptText = document.getElementById('filePromptText');
const extractBtn = document.getElementById('extractBtn');
const extractBtnText = document.getElementById('extractBtnText');
const downloadBtn = document.getElementById('downloadBtn');
const downloadBtnText = document.getElementById('downloadBtnText');
const statusBox = document.getElementById('status');
const statusText = document.getElementById('statusText');
const resultsBody = document.getElementById('resultsBody');
const rowCountBadge = document.getElementById('rowCountBadge');

let currentRows = [];
let currentBaseFileName = 'lithology_results';
let currentUnit = 'm';
let lastSavedPath = '';

const filePickerWrapper = document.getElementById('filePickerWrapper');
const clearFileBtn = document.getElementById('clearFileBtn');

function setStatus(message, type = 'info') {
  statusBox.className = `status-banner ${type}`;
  statusText.textContent = message;
}

function resetFileSelection() {
  fileInput.value = '';
  filePromptText.textContent = 'Choose well log PDF...';
  currentBaseFileName = 'lithology_results';
  if (filePickerWrapper) filePickerWrapper.classList.remove('has-file');
  currentRows = [];
  renderRows([]);
  if (downloadBtn) downloadBtn.disabled = true;
  setStatus('File removed. Select a PDF well log file to begin automated extraction.', 'info');
}

if (clearFileBtn) {
  clearFileBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    resetFileSelection();
  });
}

// Update file prompt label on file selection
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    filePromptText.textContent = file.name;
    currentBaseFileName = file.name.replace(/\.[^/.]+$/, '');
    if (filePickerWrapper) filePickerWrapper.classList.add('has-file');
    setStatus(`Selected "${file.name}". Click "Extract Lithology" to begin.`, 'info');
  } else {
    resetFileSelection();
  }
});

function formatConfidence(conf) {
  if (conf === null || conf === undefined || conf === '') return '—';
  const num = parseFloat(conf);
  if (isNaN(num)) return conf;
  const pct = (num <= 1.0 ? num * 100 : num).toFixed(1) + '%';
  let pillClass = 'conf-med';
  if (num >= 0.8 || (num > 1 && num >= 80)) pillClass = 'conf-high';
  else if (num < 0.5 || (num > 1 && num < 50)) pillClass = 'conf-low';
  return `<span class="conf-pill ${pillClass}">${pct}</span>`;
}

function renderRows(rows) {
  if (!rows || rows.length === 0) {
    resultsBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">
          <div class="empty-state">
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
            <p>No lithology intervals detected in this document.</p>
          </div>
        </td>
      </tr>
    `;
    rowCountBadge.textContent = '0 intervals';
    return;
  }

  rowCountBadge.textContent = `${rows.length} interval${rows.length === 1 ? '' : 's'}`;

  resultsBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td class="mono-num">${row.depth_start ?? '—'}</td>
          <td class="mono-num">${row.depth_end ?? '—'}</td>
          <td class="litho-name">${row.lithology ?? '—'}</td>
          <td>${formatConfidence(row.confidence)}</td>
          <td><span class="method-tag">${row.method ?? '—'}</span></td>
          <td>${row.runner_up ? `<span class="method-tag">${row.runner_up}</span>` : '—'}</td>
        </tr>
      `
    )
    .join('');
}

// Download Excel function
async function triggerExcelDownload(rows, baseName, showInExplorer = true) {
  if (!rows || rows.length === 0) {
    setStatus('No lithology data available to export.', 'error');
    return;
  }

  const filename = `${baseName || 'lithology'}_extracted.xlsx`;
  const originalHtml = downloadBtnText.textContent;
  downloadBtnText.textContent = 'Saving to PC Downloads...';
  downloadBtn.disabled = true;

  try {
    const response = await fetch('/api/export-excel', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        rows: rows,
        unit: currentUnit,
        filename: filename,
        show_in_folder: showInExplorer,
      }),
    });

    if (!response.ok) {
      let errMsg = 'Failed to generate Excel file.';
      try {
        const errJson = await response.json();
        if (errJson && errJson.error) errMsg = errJson.error;
      } catch (_) { }
      throw new Error(errMsg);
    }

    const savedPathHeader = response.headers.get('X-Saved-Path');
    if (savedPathHeader) {
      lastSavedPath = savedPathHeader;
    }

    // Trigger browser download mechanism
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const tempLink = document.createElement('a');
    tempLink.style.display = 'none';
    tempLink.href = downloadUrl;
    tempLink.download = filename;
    document.body.appendChild(tempLink);
    tempLink.click();

    // Prevent premature revocation so Chromium/Edge has time to finish saving to PC Downloads
    setTimeout(() => {
      if (tempLink.parentNode) tempLink.parentNode.removeChild(tempLink);
      window.URL.revokeObjectURL(downloadUrl);
    }, 60000);

    return true;
  } catch (err) {
    console.error('Download Excel error:', err);
    setStatus(`Excel export failed: ${err.message}`, 'error');
    return false;
  } finally {
    downloadBtnText.textContent = originalHtml;
    downloadBtn.disabled = currentRows.length === 0;
  }
}

// Extract button handler
extractBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];

  if (!file) {
    setStatus('Please choose a PDF well log file first.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  setStatus('Extracting lithology intervals. Analyzing PDF drawings and vectors...', 'working');
  extractBtn.disabled = true;
  extractBtnText.textContent = 'Processing PDF...';
  downloadBtn.disabled = true;

  try {
    const response = await fetch('/api/extract', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Extraction failed.');
    }

    currentUnit = data.unit || 'm';
    const thStart = document.getElementById('thStart');
    const thEnd = document.getElementById('thEnd');
    if (thStart) thStart.textContent = `Depth Start (${currentUnit})`;
    if (thEnd) thEnd.textContent = `Depth End (${currentUnit})`;

    currentRows = data.rows || [];
    renderRows(currentRows);

    if (currentRows.length > 0) {
      downloadBtn.disabled = false;
      setStatus(
        `✓ Extraction complete! Found ${data.row_count ?? currentRows.length} lithology intervals (depths in ${currentUnit}). Click "Download Excel (.xlsx)" to download.`,
        'success'
      );
    } else {
      setStatus('Extraction finished, but no lithology intervals were identified.', 'info');
    }
  } catch (error) {
    setStatus(error.message, 'error');
    currentRows = [];
    renderRows([]);
    downloadBtn.disabled = true;
  } finally {
    extractBtn.disabled = false;
    extractBtnText.textContent = 'Extract Lithology';
  }
});

// Download button handler (downloads only when clicked)
downloadBtn.addEventListener('click', async () => {
  if (currentRows.length === 0) {
    setStatus('Please extract lithology first before downloading.', 'error');
    return;
  }
  setStatus('Saving Excel spreadsheet to PC Downloads...', 'working');
  const downloaded = await triggerExcelDownload(currentRows, currentBaseFileName, true);
  if (downloaded) {
    setStatus(
      `✓ Excel file saved to PC Downloads (${lastSavedPath || 'Downloads'}) and shown in Windows Explorer.`,
      'success'
    );
  }
});
