const statusEl = document.getElementById('status');
const analysisSection = document.getElementById('analysis-section');
const suggestionSection = document.getElementById('suggestion-section');

function getRule(prefix) {
  return {
    text: document.getElementById(`${prefix}-text`).value.trim(),
    jurisdiction: document.getElementById(`${prefix}-jurisdiction`).value,
    statute: document.getElementById(`${prefix}-statute`).value.trim(),
    section: document.getElementById(`${prefix}-section`).value.trim(),
    enactment_year: Number(document.getElementById(`${prefix}-year`).value || 2000)
  };
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? '#dc2626' : '#4b5563';
}

function renderAnalysis(payload) {
  const { conflict_scores, components, interpretation } = payload;
  document.getElementById('score-textualist').value = conflict_scores.textualist ?? 0;
  document.getElementById('score-living').value = conflict_scores.living ?? 0;
  document.getElementById('score-pragmatic').value = conflict_scores.pragmatic ?? 0;

  const componentList = document.getElementById('component-list');
  componentList.innerHTML = '';
  Object.entries(components).forEach(([name, value]) => {
    const item = document.createElement('div');
    item.textContent = `${name}: ${value.toFixed(3)}`;
    componentList.appendChild(item);
  });
  document.getElementById('interpretation').textContent = interpretation || '';
}

function renderSuggestion(data) {
  if (!data || !data.best) {
    suggestionSection.classList.add('empty');
    document.getElementById('suggestion-body').textContent = 'No amendment suggested.';
    return;
  }
  const { best } = data;
  suggestionSection.classList.remove('empty');
  const container = document.getElementById('suggestion-body');
  container.innerHTML = '';

  const text = document.createElement('div');
  text.innerHTML = `<strong>Proposed condition:</strong> ${best.condition}`;
  container.appendChild(text);

  const justification = document.createElement('div');
  justification.innerHTML = '<strong>Justification</strong>';
  const list = document.createElement('ul');
  Object.entries(best.justification || {}).forEach(([key, value]) => {
    const li = document.createElement('li');
    li.textContent = `${key}: ${value.toFixed(3)}`;
    list.appendChild(li);
  });
  justification.appendChild(list);
  container.appendChild(justification);

  const ccs = document.createElement('div');
  ccs.innerHTML = `<strong>Estimated CCS:</strong> ${best.estimated_ccs.toFixed(3)}`;
  container.appendChild(ccs);

  const amendment = document.createElement('div');
  amendment.innerHTML = `<strong>Suggested text:</strong> ${best.suggested_text}`;
  container.appendChild(amendment);
}

async function analyze() {
  setStatus('Analyzing conflict…');
  const body = {
    rule1: getRule('rule1'),
    rule2: getRule('rule2')
  };
  try {
    const response = await fetch('/v1/law/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Analysis failed');
    }
    renderAnalysis(data);
    setStatus('Analysis complete.');
  } catch (err) {
    console.error(err);
    setStatus(err.message, true);
  }
}

async function suggest() {
  setStatus('Requesting amendment…');
  const body = {
    rule1: getRule('rule1'),
    rule2: getRule('rule2')
  };
  try {
    const response = await fetch('/v1/law/suggest_amendment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Suggestion failed');
    }
    renderSuggestion(data);
    setStatus('Suggestion ready.');
  } catch (err) {
    console.error(err);
    setStatus(err.message, true);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('rule1-text').value = 'Organizations MUST collect personal data IF consent.';
  document.getElementById('rule2-text').value = 'Security agencies MUST NOT collect personal data IF emergency.';
  document.getElementById('analyze-btn').addEventListener('click', analyze);
  document.getElementById('suggest-btn').addEventListener('click', suggest);
});
