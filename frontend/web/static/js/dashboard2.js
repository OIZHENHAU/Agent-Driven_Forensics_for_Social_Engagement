//Dashboard 2 page initilalization
document.querySelectorAll('input[type="range"]').forEach(slider => {
  const valEl = document.getElementById(slider.id + '_val');

  function refresh() {
    const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #4f6ef7 ${pct}%, #d1d5db ${pct}%)`;
    if (valEl) {
      const step = parseFloat(slider.step || 1);
      valEl.textContent = step >= 1
        ? parseInt(slider.value).toString()
        : parseFloat(slider.value).toFixed(2);
    }
  }

  slider.addEventListener('input', refresh);
  refresh();
});


//Switch the option between accoutn and post detectipon
function switchTab(mode) {
  ['account', 'post'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === mode);
    document.getElementById('section-' + t).classList.toggle('hidden', t !== mode);
  });
  document.getElementById('results').classList.add('hidden');
  document.getElementById('errorBox').classList.add('hidden');
  document.getElementById('loading').classList.add('hidden');
}


//Calculate the lexical diversity
function calculateLexicalDiversity(text) {
  if (!text || text.trim() === '') return 0;
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length === 0) return 0;
  return new Set(words).size / words.length;
}

document.getElementById('post_content').addEventListener('input', function () {
  const ld = calculateLexicalDiversity(this.value);
  document.getElementById('lexical_diversity_live').textContent = ld.toFixed(3);
});


//Tajke the user's account manual input
async function analyseAccount() {
  const btn = document.getElementById('analyseBtn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const errorBox = document.getElementById('errorBox');

  const payload = {
    followers: parseInt(document.getElementById('followers').value) || 0,
    follows: parseInt(document.getElementById('follows').value) || 0,
    posts: parseInt(document.getElementById('posts').value) || 0,
    description_length: parseInt(document.getElementById('description_length').value) || 0,
    profile_pic: parseInt(document.getElementById('profile_pic').value),
    private: parseInt(document.getElementById('private').value),
    external_url: parseInt(document.getElementById('external_url').value),
    name_equals_username: parseInt(document.getElementById('name_equals_username').value),
    fullname_words:  parseInt(document.getElementById('fullname_words').value) || 1,
    nums_length_username: parseFloat(document.getElementById('nums_length_username').value),
    nums_length_fullname: parseFloat(document.getElementById('nums_length_fullname').value),
  };

  btn.disabled = true;
  results.classList.add('hidden');
  errorBox.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const res  = await fetch('/api/analyze-account', {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    display_results(data, 'account');

  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    loading.classList.add('hidden');
  }
}


//Take the user's post manual input
async function analysePost() {
  const btn = document.getElementById('analysePostBtn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const errorBox = document.getElementById('errorBox');

  const content = document.getElementById('post_content').value.trim();
  const dayOfWeek = parseInt(document.getElementById('day_of_week').value);

  const payload = {
    content,
    character_count: content.length,
    likes: parseInt(document.getElementById('post_likes').value) || 0,
    comments: parseInt(document.getElementById('post_comments').value) || 0,
    shares: parseInt(document.getElementById('post_shares').value) || 0,
    hashtag_count: parseInt(document.getElementById('hashtag_count').value) || 0,
    mention_count: parseInt(document.getElementById('mention_count').value) || 0,
    contains_url: parseInt(document.getElementById('post_contains_url').value),
    has_media: parseInt(document.getElementById('has_media').value),
    hour_of_day: parseInt(document.getElementById('hour_of_day').value),
    day_of_week: dayOfWeek,
    is_weekend: [5, 6].includes(dayOfWeek) ? 1 : 0,
  };

  btn.disabled = true;
  results.classList.add('hidden');
  errorBox.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const res  = await fetch('/api/analyze-post', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    display_results(data, 'post');

  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove('hidden');

  } finally {
    btn.disabled = false;
    loading.classList.add('hidden');

  }
}



function display_results(data, mode) {
  const { if_score, lof_score, ensemble_score, verdict, gemini_explanation, lexical_diversity } = data;
  const suspicious = ensemble_score < 50;

  // Lexical diversity part
  const lexicalDiversitySection = document.getElementById('ld_section');

  if (mode === 'post' && lexical_diversity !== undefined) {
    document.getElementById('ld_result_score').textContent = typeof lexical_diversity === 'number' ? lexical_diversity.toFixed(3) : lexical_diversity;
    lexicalDiversitySection.classList.remove('hidden');

  } else {
    lexicalDiversitySection.classList.add('hidden');

  }

  setScore('if_score', 'if_status', if_score, if_score < 50);
  setScore('lof_score', 'lof_status', lof_score, lof_score < 50);
  setScore('ensemble_score', 'ensemble_status', ensemble_score, suspicious);

  const banner  = document.getElementById('verdict_banner');
  banner.className = `verdict-banner ${suspicious ? 'suspicious' : 'authentic'}`;

  const subject = mode === 'post' ? 'post' : 'account';
  document.getElementById('verdict_title').textContent = suspicious ? `Suspicious ${subject}` : `Authentic ${subject}`;

  document.getElementById('verdict_text').textContent = suspicious
    ? `Ensemble authenticity score: ${ensemble_score} / 100 — ${subject} metrics appear anomalous`
    : `Ensemble authenticity score: ${ensemble_score} / 100 — ${subject} metrics appear genuine`;

  const geminiBody = document.getElementById('gemini_body');

  if (gemini_explanation) {
    geminiBody.innerHTML = gemini_explanation.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                                              .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');

    document.getElementById('gemini_section').style.display = '';

  } else {
    document.getElementById('gemini_section').style.display = 'none';

  }

  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}


function setScore(modelId, statusId, score, isBadCondition) {
  const numOfElements = document.getElementById(modelId);
  const statusOfElements = document.getElementById(statusId);
  numOfElements.textContent = score;
  numOfElements.className = `score-num${isBadCondition ? '' : ' authentic'}`;
  statusOfElements.textContent = isBadCondition ? 'Suspicious' : 'Authentic';
  statusOfElements.className = `score-status${isBadCondition ? '' : ' authentic'}`;
}
