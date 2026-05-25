document.querySelectorAll('input[type="range"]').forEach(slider => {
  const slide_element = document.getElementById(slider.id + '_val');

  function refresh() {
    const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #4f6ef7 ${pct}%, #d1d5db ${pct}%)`;
    if (slide_element) slide_element.textContent = parseFloat(slider.value).toFixed(2);
  }

  slider.addEventListener('input', refresh);
  refresh();

});


async function analyseAccount() {
  const btn = document.getElementById('analyseBtn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const errorBox = document.getElementById('errorBox');

  const payload = {
    followers: parseInt(document.getElementById('followers').value) || 0,
    follows: parseInt(document.getElementById('follows').value) || 0,
    posts: parseInt(document.getElementById('posts').value) || 0,
    description_length:  parseInt(document.getElementById('description_length').value)  || 0,
    profile_pic: parseInt(document.getElementById('profile_pic').value),
    private: parseInt(document.getElementById('private').value),
    external_url: parseInt(document.getElementById('external_url').value),
    name_equals_username: parseInt(document.getElementById('name_equals_username').value),
    fullname_words: parseInt(document.getElementById('fullname_words').value) || 1,
    nums_length_username : parseFloat(document.getElementById('nums_length_username').value),
    nums_length_fullname : parseFloat(document.getElementById('nums_length_fullname').value),
  };

  btn.disabled = true;
  results.classList.add('hidden');
  errorBox.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const res = await fetch('/api/analyze-account', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    render_results(data);

  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove('hidden');

  } finally {
    btn.disabled = false;
    loading.classList.add('hidden');
  }
}


function render_results(data) {
  const { if_score, lof_score, ensemble_score, verdict, gemini_explanation } = data;
  const suspicious = ensemble_score < 50;

  setScore('if_score', 'if_status', if_score, if_score < 50);
  setScore('lof_score', 'lof_status', lof_score, lof_score < 50);
  setScore('ensemble_score', 'ensemble_status', ensemble_score, suspicious);

  const banner = document.getElementById('verdict_banner');
  banner.className = `verdict-banner ${suspicious ? 'suspicious' : 'authentic'}`;

  document.getElementById('verdict_title').textContent = suspicious
    ? 'Suspicious account'
    : 'Authentic account';

  document.getElementById('verdict_text').textContent = suspicious
    ? `Ensemble authenticity score: ${ensemble_score} / 100 — both models agree this account is anomalous`
    : `Ensemble authenticity score: ${ensemble_score} / 100 — account metrics appear genuine`;

  const geminiBody = document.getElementById('gemini_body');
  if (gemini_explanation) {
    geminiBody.innerHTML = gemini_explanation
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
    document.getElementById('gemini_section').style.display = '';
  } else {
    document.getElementById('gemini_section').style.display = 'none';
  }

  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setScore(numId, statusId, score, isBad) {
  const numEl = document.getElementById(numId);
  const statusEl = document.getElementById(statusId);

  numEl.textContent = score;
  numEl.className = `score-num${isBad ? '' : ' authentic'}`;

  statusEl.textContent = isBad ? 'Suspicious' : 'Authentic';
  statusEl.className = `score-status${isBad ? '' : ' authentic'}`;
}
