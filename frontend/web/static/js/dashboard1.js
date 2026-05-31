let globalModelResults = null;
let currentTab = 'post';
const chartInstances = {};

const PLOT_LABELS = {
  'Isolation_Forest_Confusion_Matrix_post': 'IF Confusion Matrix',
  'LOF_Confusion_Matrix_post': 'LOF Confusion Matrix',
  'Isolation_Forest_Result_post': 'IF Anomaly Results',
  'LOF_Result_post': 'LOF Anomaly Results',
  'Feature_Importance_IF_post': 'Feature Importance',
  'pca_projection_post': 'PCA Projection',
  'pca_variance_post': 'PCA Variance',
  'all_histograms_features_post': 'Feature Histograms',
  'all_boxplots_features_post': 'Feature Boxplots',
  'correlation_heatmap_post': 'Correlation Heatmap',
  'all_log_validation_features_post': 'Log Validation',
  'Isolation_Forest_Confusion_Matrix': 'IF Confusion Matrix',
  'LOF_Confusion_Matrix': 'LOF Confusion Matrix',
  'Isolation_Forest_Result': 'IF Anomaly Results',
  'LOF_Result': 'LOF Anomaly Results',
  'pca_projection_account': 'PCA Projection',
  'pca_variance_account': 'PCA Variance',
  'all_histograms_features_account': 'Feature Histograms',
  'all_boxplots_features_account': 'Feature Boxplots',
  'correlation_heatmap_account': 'Correlation Heatmap',
};

function getPlotLabel(filename) {
  const stem = filename.replace(/\.png$/i, '');
  return PLOT_LABELS[stem] || stem.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

//Initialize the dashnboard 1 page.
window.addEventListener('DOMContentLoaded', async () => {
  await loadModelsResult();
  switchTab('post');
});

//Switch between each dashoborad
function switchTab(mode) {
  currentTab = mode;
  ['post', 'account'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === mode);
    document.getElementById('section-' + t).classList.toggle('hidden', t !== mode);
  });
}

//LOad the model result with two models
async function loadModelsResult() {
  try {
    const res = await fetch('/api/get_model_result');
    if (res.status === 404) {
      document.getElementById('loading-section').classList.add('hidden');
      document.getElementById('not-ready').classList.remove('hidden');
      return;
    }
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    globalModelResults = data;

    displayModelResultCard('post-if-card', 'Isolation Forest', data.post_detection?.isolation_forest);
    displayModelResultCard('post-lof-card', 'Local Outlier Factor', data.post_detection?.lof);
    displayModelResultCard('account-if-card', 'Isolation Forest', data.account_detection?.isolation_forest);
    displayModelResultCard('account-lof-card', 'Local Outlier Factor', data.account_detection?.lof);

    displayModelResultBarChart('post-chart', data.post_detection);
    displayModelResultBarChart('account-chart', data.account_detection);

    await loadPlots('post');
    await loadPlots('account');

    document.getElementById('loading-section').classList.add('hidden');
    document.getElementById('section-post').classList.remove('hidden');
    document.getElementById('report-section').style.display = '';

  } catch (err) {
    document.getElementById('loading-section').classList.add('hidden');
    const box = document.getElementById('not-ready');
    box.textContent = 'Error loading model results: ' + err.message;
    box.classList.remove('hidden');
  }
}


function displayModelResultCard(cardId, title, d = {}) {
  const keys   = ['accuracy', 'precision', 'recall', 'f1'];
  const labels = { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1 Score' };

  const rows = keys.map(k => {
    const v   = d[k] ?? 0;
    const pct = (v * 100).toFixed(1) + '%';
    const cls = v >= 0.7 ? 'good' : v >= 0.5 ? 'ok' : 'poor';
    return `<div class="model-results-row">
      <span class="model-results-name">${labels[k]}</span>
      <span class="model-results-value ${cls}">${pct}</span>
    </div>`;
  }).join('');

  document.getElementById(cardId).innerHTML = `<div class="model-results-card-title">${title}</div>${rows}`;

}


//Display the bar chart for the accuracy, precision, and recall for the tewo model 
function displayModelResultBarChart(canvasId, detection = {}) {
  const if_d  = detection.isolation_forest || {};
  const lof_d = detection.lof || {};
  const labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
  const keys = ['accuracy', 'precision', 'recall', 'f1'];

  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  chartInstances[canvasId] = new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Isolation Forest',
          data: keys.map(k => parseFloat(((if_d[k] || 0) * 100).toFixed(1))),
          backgroundColor: 'rgba(79, 110, 247, 0.75)',
          borderRadius: 6,
        },
        {
          label: 'Local Outlier Factor',
          data: keys.map(k => parseFloat(((lof_d[k] || 0) * 100).toFixed(1))),
          backgroundColor: 'rgba(168, 85, 247, 0.75)',
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 12 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%` } },
      },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + '%' },
          grid: { color: '#f3f4f6' },
        },
        x: { grid: { display: false } },
      },
    },
  });
  
}


//Load all the plots and diagram from the outputs path
async function loadPlots(category) {
  const selectorId = category + '-plot-selector';
  const selector = document.getElementById(selectorId);

  try {
    const res = await fetch(`/api/output-images/${category}`);
    const data = await res.json();
    const images = data.images || [];

    if (!images.length) {
      selector.innerHTML = '<p class="plot-hint">No plots found. Run the pipeline first.</p>';
      return;
    }

    selector.innerHTML = images.map(img =>
      `<button class="plot-btn" onclick="showPlot('${category}','${img}',this)">${getPlotLabel(img)}</button>`).join('');

  } catch {
    selector.innerHTML = '<p class="plot-hint">Could not load plots.</p>';
  }
}

//Display the plot and the diagram
function showPlot(category, filename, button) {
  // Update active button
  const selector = document.getElementById(category + '-plot-selector');
  selector.querySelectorAll('.plot-btn').forEach(b => b.classList.remove('active'));
  button.classList.add('active');

  // Show image
  const image = document.getElementById(category + '-plot-img');
  const placeholder = document.getElementById(category + '-plot-placeholder');

  image.src = `/outputs/${category}/${filename}`;
  image.alt = getPlotLabel(filename);
  image.classList.remove('hidden');
  placeholder.classList.add('hidden');

}


//Genaerate an AI summary resport
async function generateReport() {
  if (!globalModelResults) {
    return;
  }

  const button = document.getElementById('report-btn');
  const loading = document.getElementById('report-loading');
  const body = document.getElementById('report-body');

  button.disabled = true;
  loading.classList.remove('hidden');
  body.classList.add('hidden');

  try {
    const res  = await fetch('/api/generate-model-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelResults: globalModelResults }),
    });
    
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    body.innerHTML = data.report.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');

    body.classList.remove('hidden');
    button.textContent = 'Regenerate Report';

  } catch (err) {
    body.innerHTML = `<span style="color:#dc2626">Error: ${err.message}</span>`;
    body.classList.remove('hidden');

  } finally {
    loading.classList.add('hidden');
    button.disabled = false;

  }

}