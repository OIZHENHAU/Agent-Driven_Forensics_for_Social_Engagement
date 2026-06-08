let currentMode = 'post';
let parsedRows = [];
let analysisResults = [];
let aiCache = {};
let activeRowIndex = null;


function switchTab(mode) {
  currentMode = mode;
  ['account', 'post'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === mode);
    document.getElementById('cols-hint-' + t).classList.toggle('hidden', t !== mode);
  });
  resetAll();
}


function resetAll() {
  parsedRows = [];
  analysisResults = [];
  aiCache = {};
  document.getElementById('file-loaded-section').classList.add('hidden');
  document.getElementById('loading-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('errorBox').classList.add('hidden');
  document.getElementById('csv-file-input').value = '';
}


document.getElementById('csv-file-input').addEventListener('change', function () {
  if (this.files && this.files[0]) handleFile(this.files[0]);
});

const uploadZone = document.getElementById('upload-zone');
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && /\.(csv|xlsx|xls)$/i.test(file.name)) handleFile(file);
});
uploadZone.addEventListener('click', e => {
  if (e.target !== document.getElementById('csv-file-input') &&
      !e.target.classList.contains('browse-btn')) return;
});


function handleFile(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const text = e.target.result;
    parsedRows = parseCsv(text);
    if (!parsedRows.length) {
      showError('CSV appears to be empty or unreadable.');
      return;
    }
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('row-count-badge').textContent = parsedRows.length + ' rows';
    renderPreview(parsedRows);
    document.getElementById('file-loaded-section').classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('loading-section').classList.add('hidden');
  };
  reader.readAsText(file);
}

function parseCsv(text) {
  const lines = text.trim().split('\n').filter(l => l.trim());
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  return lines.slice(1).map(line => {
    const vals = splitCsvLine(line);
    const row = {};
    headers.forEach((h, i) => { row[h] = (vals[i] || '').trim().replace(/^"|"$/g, ''); });
    return row;
  });
}

function splitCsvLine(line) {
  const result = [];
  let cur = '', inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQuote = !inQuote; }
    else if (ch === ',' && !inQuote) { result.push(cur); cur = ''; }
    else { cur += ch; }
  }
  result.push(cur);
  return result;
}

function renderPreview(rows) {
  const table = document.getElementById('preview-table');
  const headers = Object.keys(rows[0]);
  const previewRows = rows.slice(0, 5);
  let html = '<thead><tr>' + headers.map(h => `<th>${esc(h)}</th>`).join('') + '</tr></thead>';
  html += '<tbody>' + previewRows.map(row =>
    '<tr>' + headers.map(h => `<td>${esc(row[h] ?? '')}</td>`).join('') + '</tr>'
  ).join('') + '</tbody>';
  table.innerHTML = html;
}


async function runAnalysis() {
  if (!parsedRows.length) return;

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  document.getElementById('errorBox').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');

  const csvText = rowsToCsv(parsedRows);
  const blob = new Blob([csvText], { type: 'text/csv' });
  const formData = new FormData();
  formData.append('file', blob, 'upload.csv');
  formData.append('type', currentMode);

  document.getElementById('loading-section').classList.remove('hidden');

  try {
    const res = await fetch('/api/analyze-csv', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    analysisResults = data.results;
    aiCache = {};
    renderResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    document.getElementById('loading-section').classList.add('hidden');
    runBtn.disabled = false;
  }
}

function rowsToCsv(rows) {
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(',')];
  rows.forEach(row => lines.push(headers.map(h => JSON.stringify(row[h] ?? '')).join(',')));
  return lines.join('\n');
}



function renderResults(data) {
  document.getElementById('sum-total').textContent = data.total;
  document.getElementById('sum-authentic').textContent = data.authentic;
  document.getElementById('sum-suspicious').textContent = data.suspicious;
  document.getElementById('results-count').textContent = data.total;

  const isPost = currentMode === 'post';
  const keyHeaders = isPost
    ? ['Likes', 'Comments', 'Shares']
    : ['Followers', 'Following', 'Posts'];

  const identifierCol = detectIdentifier(data.results[0]?.row_data || {});
  const headers = ['#', ...(identifierCol ? [identifierCol] : []), ...keyHeaders,
                   'IF Score', 'LOF Score', 'Ensemble', 'Verdict'];

  document.getElementById('results-thead').innerHTML =
    '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';

  document.getElementById('results-tbody').innerHTML = data.results.map((r, i) => {
    const rd = r.row_data || {};
    const suspicious = r.ensemble_score < 50;
    const scoreColor = suspicious ? '#d94f4f' : '#16a34a';
    const keyVals = isPost
      ? [rd.likes ?? rd.like_count ?? '—', rd.comments ?? rd.comment_count ?? '—', rd.shares ?? rd.share_count ?? '—']
      : [rd.followers ?? rd['#followers'] ?? '—',
         rd.following ?? rd.follows ?? rd['#follows'] ?? '—',
         rd.posts ?? rd['#posts'] ?? '—'];
    const idCell = identifierCol ? `<td>${esc(String(rd[identifierCol] ?? '—'))}</td>` : '';
    const badge  = `<span class="badge ${suspicious ? 'badge-suspicious' : 'badge-authentic'}">${r.verdict}</span>`;

    return `<tr onclick="openModal(${i})">
      <td>${i + 1}</td>
      ${idCell}
      ${keyVals.map(v => `<td>${esc(String(v))}</td>`).join('')}
      <td style="color:${scoreColor};font-weight:600">${r.if_score}</td>
      <td style="color:${scoreColor};font-weight:600">${r.lof_score}</td>
      <td style="color:${scoreColor};font-weight:700">${r.ensemble_score}</td>
      <td>${badge}</td>
    </tr>`;
  }).join('');

  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function detectIdentifier(rowData) {
  const candidates = ['username', 'name', 'user', 'id', 'account', 'handle', 'post_id', 'activity_id'];
  for (const c of candidates) {
    if (rowData[c] !== undefined) return c;
  }
  return null;
}


function openModal(index) {
  activeRowIndex = index;
  const r = analysisResults[index];
  const rd = r.row_data || {};
  const suspicious = r.ensemble_score < 50;
  const isPost = currentMode === 'post';

  document.getElementById('modal-row-label').textContent = `Row #${index + 1}`;

  const keyPairs = isPost
    ? [['Likes', rd.likes ?? '—'], ['Comments', rd.comments ?? '—'],
       ['Shares', rd.shares ?? '—'], ['Hashtags', rd.hashtag_count ?? '—'],
       ['Lex. Diversity', r.lexical_diversity ?? '—']]
    : [['Followers', rd.followers ?? rd['#followers'] ?? '—'],
       ['Following', rd.following ?? rd.follows ?? rd['#follows'] ?? '—'],
       ['Posts', rd.posts ?? rd['#posts'] ?? '—'],
       ['Bio length', rd.description_length ?? rd['description length'] ?? '—']];

  const identifierCol = detectIdentifier(rd);
  const idPart = identifierCol
    ? `<strong>${esc(identifierCol)}:</strong> ${esc(String(rd[identifierCol]))} &nbsp;&middot;&nbsp; `
    : '';
  document.getElementById('modal-key-data').innerHTML =
    idPart + keyPairs.map(([k, v]) => `<strong>${k}:</strong> ${esc(String(v))}`).join(' &nbsp;&middot;&nbsp; ');

  setModalScore('m-if-score', 'm-if-status', r.if_score, r.if_score < 50);
  setModalScore('m-lof-score', 'm-lof-status', r.lof_score, r.lof_score < 50);
  setModalScore('m-ensemble-score', 'm-ensemble-status', r.ensemble_score, suspicious);

  const banner = document.getElementById('m-verdict-banner');
  banner.className = `verdict-banner ${suspicious ? 'suspicious' : 'authentic'}`;
  document.getElementById('m-verdict-title').textContent =
    (suspicious ? 'Suspicious ' : 'Authentic ') + (isPost ? 'post' : 'account');
  document.getElementById('m-verdict-text').textContent =
    `Ensemble score: ${r.ensemble_score} / 100 — ${suspicious ? 'metrics appear anomalous' : 'metrics appear genuine'}`;

  const aiBtn = document.getElementById('ai-btn');
  aiBtn.disabled = false;
  aiBtn.textContent = 'Analyze';
  document.getElementById('m-loading').classList.add('hidden');
  document.getElementById('m-gemini').classList.add('hidden');

  if (aiCache[index]) {
    showGeminiText(aiCache[index]);
    aiBtn.textContent = 'Reload Analyze';
  }

  document.getElementById('modal-overlay').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function setModalScore(numId, statusId, score, isBad) {
  const numEl    = document.getElementById(numId);
  const statusEl = document.getElementById(statusId);
  numEl.textContent    = score;
  numEl.className      = `score-num${isBad ? '' : ' authentic'}`;
  statusEl.textContent = isBad ? 'Suspicious' : 'Authentic';
  statusEl.className   = `score-status${isBad ? '' : ' authentic'}`;
}

function closeModalDirect() {
  document.getElementById('modal-overlay').classList.add('hidden');
  document.body.style.overflow = '';
}

document.getElementById('modal-overlay').addEventListener('click', function (e) {
  if (e.target === this) closeModalDirect();
});
document.getElementById('modal-close-btn').addEventListener('click', closeModalDirect);



async function loadAiAnalysis() {
  if (activeRowIndex === null) return;

  if (aiCache[activeRowIndex]) {
    showGeminiText(aiCache[activeRowIndex]);
    return;
  }

  const r = analysisResults[activeRowIndex];
  const aiBtn = document.getElementById('ai-btn');
  aiBtn.disabled = true;
  document.getElementById('m-loading').classList.remove('hidden');
  document.getElementById('m-gemini').classList.add('hidden');

  try {
    const res = await fetch('/api/analyze-csv-row', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: currentMode,
        row_data: r.row_data,
        scores: {
          if_score: r.if_score,
          lof_score: r.lof_score,
          ensemble_score: r.ensemble_score,
          verdict: r.verdict,
          lexical_diversity: r.lexical_diversity,
        },
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    const text = data.gemini_explanation || '(No analysis returned)';
    aiCache[activeRowIndex] = text;
    showGeminiText(text);

  } catch (err) {
    showGeminiText(`AI analysis unavailable: ${err.message}`);
  } finally {
    document.getElementById('m-loading').classList.add('hidden');
    aiBtn.disabled = false;
    aiBtn.textContent = 'Reload Analyze';
  }
}

function showGeminiText(text) {
  const cleaned = text.replace(/```[\s\S]*?```/g, '').trim();
  const body = document.getElementById('m-gemini-body');
  body.innerHTML = cleaned
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  document.getElementById('m-gemini').classList.remove('hidden');
}


function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = 'Error: ' + msg;
  box.classList.remove('hidden');
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
