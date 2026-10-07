/**
 * Diagnostic Odyssey AI — Application Logic
 * Matches the royal editorial HTML redesign
 */

let currentPosts = [];
let sampleCases  = {};
let chartInst    = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadSampleCases();
  loadBenchmarkData();
});

/* ── Tab Navigation ─────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const t = document.getElementById(btn.dataset.tab);
      if (t) t.classList.add('active');
    });
  });
}

/* ── Sample Cases ───────────────────────────────── */
async function loadSampleCases() {
  try {
    const res  = await fetch('/api/sample_cases');
    sampleCases = await res.json();
    loadPreset('case_1_lupus_long_delay');
  } catch {
    currentPosts = [
      { elapsed_days: 0,   text: 'Dealing with debilitating fatigue, low-grade fevers, and morning stiffness in my hands for months. Doctors say all basic bloodwork is normal.' },
      { elapsed_days: 180, text: 'Saw another doctor today. He told me it is all in my head and just anxiety from stress at work. Refused to order an autoimmune panel.' },
      { elapsed_days: 520, text: 'Now developed a severe malar butterfly rash across my cheeks after sun exposure and burning neuropathy in my feet. How do you find a doctor who actually listens?' }
    ];
    renderPosts();
  }
}

function loadPreset(key) {
  if (!sampleCases[key]) return;
  currentPosts = JSON.parse(JSON.stringify(sampleCases[key].posts));
  renderPosts();
  runPrediction();
}

/* ── Posts ──────────────────────────────────────── */
function renderPosts() {
  const container = document.getElementById('posts-container');
  container.innerHTML = '';

  currentPosts.forEach((post, idx) => {
    const div = document.createElement('div');
    div.className = 'post-card';
    div.id = `post-card-${idx}`;
    div.innerHTML = `
      <div class="post-card-top">
        <span class="post-num">Encounter ${idx + 1}</span>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:0.72rem;color:var(--cream-muted);">Day&nbsp;</span>
          <input type="number" class="post-days"
                 value="${post.elapsed_days}"
                 onchange="updateDays(${idx}, this.value)">
          <button class="post-del" onclick="deletePost(${idx})" title="Remove">×</button>
        </div>
      </div>
      <textarea class="post-text"
                onchange="updateText(${idx}, this.value)">${post.text}</textarea>
    `;
    container.appendChild(div);
  });
}

function updateDays(idx, v) { if (currentPosts[idx]) currentPosts[idx].elapsed_days = parseFloat(v) || 0; }
function updateText(idx, v) { if (currentPosts[idx]) currentPosts[idx].text = v; }

function addPost() {
  const last = currentPosts.length > 0 ? currentPosts[currentPosts.length - 1].elapsed_days + 60 : 0;
  currentPosts.push({ elapsed_days: last, text: 'New encounter: symptoms persisting, awaiting specialist follow-up.' });
  renderPosts();
}

function deletePost(idx) {
  if (currentPosts.length <= 1) return;
  currentPosts.splice(idx, 1);
  renderPosts();
}

/* ── Run Prediction ─────────────────────────────── */
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  btn.textContent = 'Computing...';
  btn.disabled = true;

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ posts: currentPosts })
    });
    const data = await res.json();

    if (data.warming_up) {
      // Model still loading in background — show message and auto-retry
      btn.textContent = 'Model is warming up — retrying in 6 seconds...';
      showWarmingBanner();
      setTimeout(() => {
        btn.textContent = 'Run Longitudinal Prediction & Explainability';
        btn.disabled = false;
        runPrediction();
      }, 6000);
      return;
    }

    hideWarmingBanner();
    displayResults(data);
  } catch (e) {
    console.error(e);
    btn.textContent = 'Connection error — is the server running?';
  } finally {
    if (!btn.disabled || btn.textContent === 'Connection error — is the server running?') {
      btn.textContent = 'Run Longitudinal Prediction & Explainability';
      btn.disabled = false;
    }
  }
}

function showWarmingBanner() {
  let b = document.getElementById('warming-banner');
  if (!b) {
    b = document.createElement('div');
    b.id = 'warming-banner';
    b.style.cssText = `
      background: rgba(201,164,96,0.08);
      border: 1px solid rgba(201,164,96,0.25);
      border-left: 3px solid #c9a460;
      color: #b8ad97;
      font-size: 0.8rem;
      padding: 10px 14px;
      border-radius: 0 5px 5px 0;
      margin-bottom: 12px;
      line-height: 1.5;
    `;
    b.textContent = 'The AI model is loading in the background (first visit after deployment takes ~25 seconds). The page is fully usable — predictions will run automatically once ready.';
    const panel = document.querySelector('.panel-section:last-child');
    if (panel) panel.insertBefore(b, panel.firstChild);
  }
}

function hideWarmingBanner() {
  const b = document.getElementById('warming-banner');
  if (b) b.remove();
}


/* ── Display Results ────────────────────────────── */
function displayResults(data) {
  if (!data?.prediction) return;

  const pred    = data.prediction;
  const metrics = data.extracted_metrics;
  const pct     = Math.round(pred.probability * 100);
  const isHigh  = pred.probability >= 0.5;

  // hide empty state, show results
  const emptyEl   = document.getElementById('empty-state');
  const resultsEl = document.getElementById('results-content');
  if (emptyEl)   emptyEl.style.display   = 'none';
  if (resultsEl) resultsEl.style.display = 'flex';

  // Dial
  const dial = document.getElementById('risk-dial');
  document.getElementById('dial-pct').textContent = `${pct}%`;
  dial.className = 'risk-dial ' + (isHigh ? 'high' : 'low');

  // Tag / headline / sub
  const tag = document.getElementById('risk-tag');
  tag.textContent  = isHigh ? 'High Delay Risk' : 'Low Delay Risk';
  tag.className    = 'risk-level-tag ' + (isHigh ? 'danger' : 'safe');

  document.getElementById('risk-headline').textContent = pred.predicted_class;
  document.getElementById('risk-sub').textContent = isHigh
    ? 'Patient shows significant medical dismissal signals, high multi-system symptom breadth, and elevated frustration slopes consistent with a prolonged diagnostic odyssey.'
    : 'Patient trajectory indicates a prompt diagnostic workup with focused symptomatology and early clinical validation — consistent with a short delay.';

  // Metrics strip
  const strip = document.getElementById('metrics-strip');
  strip.style.display = 'flex';
  document.getElementById('m-span').textContent  = `${metrics.total_span_days}d`;
  document.getElementById('m-gap').textContent   = `${metrics.avg_gap_days}d`;
  document.getElementById('m-dis').textContent   = `${metrics.dismissal_count}`;
  document.getElementById('m-sys').textContent   = `${metrics.symptom_breadth}`;
  document.getElementById('m-frust').textContent = `${metrics.frustration_slope >= 0 ? '+' : ''}${metrics.frustration_slope}`;

  // Attention chart
  const attnSec = document.getElementById('attention-section');
  attnSec.style.display = 'block';
  renderAttentionChart(data.timeline);

  // Turning point callout
  const tp = data.timeline.find(t => t.is_turning_point);
  const callout = document.getElementById('turning-point-callout');
  if (tp) {
    const snippet = tp.text.length > 120 ? tp.text.slice(0, 120) + '...' : tp.text;
    callout.innerHTML = `<strong>Diagnostic Milestone — Encounter ${tp.post_index}, Day ${tp.elapsed_days}:</strong> The model assigned peak attention (${Math.round(tp.attention_weight * 100)}%) to this post: <em>"${snippet}"</em>`;
  }

  // Highlight turning point in timeline editor
  document.querySelectorAll('.post-card').forEach(el => el.classList.remove('is-peak'));
  if (tp) {
    const pk = document.getElementById(`post-card-${tp.post_index - 1}`);
    if (pk) pk.classList.add('is-peak');
  }

  // SHAP
  const shapSec = document.getElementById('shap-section');
  shapSec.style.display = 'block';
  renderSHAP(data.top_shap_features);
}

/* ── Attention Chart ────────────────────────────── */
function renderAttentionChart(timeline) {
  const ctx    = document.getElementById('attentionChart').getContext('2d');
  const labels = timeline.map(t => `Post ${t.post_index}  (Day ${Math.round(t.elapsed_days)})`);
  const values = timeline.map(t => t.attention_weight);
  const colors = timeline.map(t => t.is_turning_point ? 'rgba(160,80,80,0.8)' : 'rgba(201,164,96,0.45)');
  const bords  = timeline.map(t => t.is_turning_point ? 'rgba(160,80,80,1)'   : 'rgba(201,164,96,0.7)');

  if (chartInst) chartInst.destroy();

  chartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Attention Weight',
        data:  values,
        backgroundColor: colors,
        borderColor:     bords,
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: c => ` α_t = ${(c.raw * 100).toFixed(1)}%` }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid:  { color: 'rgba(201,164,96,0.07)' },
          ticks: { color: '#7a7060', font: { family: 'JetBrains Mono', size: 11 } }
        },
        x: {
          grid:  { display: false },
          ticks: { color: '#7a7060', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

/* ── SHAP Bars ──────────────────────────────────── */
function renderSHAP(features) {
  const container = document.getElementById('shap-bars');
  container.innerHTML = '';
  if (!features?.length) return;

  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_impact))) || 1;

  features.forEach(f => {
    const pos  = f.shap_impact > 0;
    const pct  = Math.min(100, Math.round((Math.abs(f.shap_impact) / maxAbs) * 100));
    const row  = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <span class="shap-key" title="${f.feature}">${f.feature}</span>
      <div class="shap-track">
        <div class="shap-fill ${pos ? 'pos' : 'neg'}" style="width:${pct}%"></div>
      </div>
      <span class="shap-val" style="color:${pos ? '#b07070' : '#6aaa88'}">
        ${pos ? '+' : ''}${f.shap_impact.toFixed(3)}
      </span>
    `;
    container.appendChild(row);
  });
}

/* ── Benchmark Table ────────────────────────────── */
async function loadBenchmarkData() {
  try {
    const res  = await fetch('/api/benchmark');
    const data = await res.json();
    const tbody = document.getElementById('bench-tbody');
    tbody.innerHTML = '';

    data.models.forEach(m => {
      const isBest = m.Model.includes('BiLSTM');
      const tr = document.createElement('tr');
      if (isBest) tr.className = 'best';
      tr.innerHTML = `
        <td>${m.Model.replace(/_/g, ' ')}${isBest ? ' &nbsp;<span style="color:var(--gold);font-size:0.72rem;">Proposed</span>' : ''}</td>
        <td>${(m.Accuracy  * 100).toFixed(1)}%</td>
        <td>${(m.Precision * 100).toFixed(1)}%</td>
        <td>${(m.Recall    * 100).toFixed(1)}%</td>
        <td>${(m.F1_Score  * 100).toFixed(1)}%</td>
        <td>${m.ROC_AUC.toFixed(3)}</td>
        <td>${m.PR_AUC.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('Benchmark load error', e);
  }
}

/* ── Anonymization ──────────────────────────────── */
async function runAnonymization() {
  const author = document.getElementById('raw-author-input').value;
  const text   = document.getElementById('raw-text-input').value;
  try {
    const res  = await fetch('/api/anonymize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ author, text })
    });
    const data = await res.json();
    document.getElementById('anon-id-output').textContent   = data.patient_id;
    document.getElementById('anon-text-output').textContent = data.scrubbed_text;
  } catch (e) {
    console.error('Anonymization error', e);
  }
}
