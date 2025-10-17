const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const api = {
  base: '',
  async request(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }
    const response = await fetch(`${api.base}${path}`, opts);
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (err) {
      data = text;
    }
    if (!response.ok) {
      const detail = data && data.detail ? data.detail : data;
      const message = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
      throw new Error(message || response.statusText);
    }
    return data;
  },
  post(path, body) {
    return api.request('POST', path, body);
  },
  get(path) {
    return api.request('GET', path);
  }
};

function setExample(kind) {
  const dsl = $('#dsl');
  const units = $('#units');
  const domain = $('#domain');
  const pdeSpace = $('#pdeSpace');
  const pdeTime = $('#pdeTime');
  $('#constraints').value = '';
  $('#resultUnit').value = '';

  if (kind === 'einstein') {
    dsl.value = 'E = m*c^2';
    units.value = 'm: kg\nc: m/s\nE: J';
    domain.value = 'classical';
    pdeSpace.value = '';
    pdeTime.value = '';
  } else if (kind === 'newton') {
    dsl.value = 'F == m*a';
    units.value = 'F: N\nm: kg\na: m/s^2';
    domain.value = 'classical';
    pdeSpace.value = '';
    pdeTime.value = '';
  } else if (kind === 'wave') {
    dsl.value = 'd2(u,t) == c**2 * d2(u,x)';
    units.value = 'c: m/s';
    domain.value = 'classical';
    pdeSpace.value = 'x';
    pdeTime.value = 't';
  } else if (kind === 'gamma') {
    dsl.value = 'gamma = 1/sqrt(1 - (v/c)**2)';
    units.value = 'v: m/s\nc: m/s';
    domain.value = 'relativistic';
    pdeSpace.value = '';
    pdeTime.value = '';
  }
  renderInput();
}

function parseKV(text) {
  const out = {};
  (text || '').split(/\r?\n/).forEach((line) => {
    const parts = line.split(':');
    if (parts.length >= 2) {
      const key = parts[0].trim();
      const value = parts.slice(1).join(':').trim();
      if (key) out[key] = value;
    }
  });
  return out;
}

function safeJSON(text) {
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (err) {
    return {};
  }
}

function resetOutputs() {
  $('#parsed').textContent = '';
  const canonical = $('#canonicalText');
  canonical.innerHTML = '';
  canonical.dataset.copy = '';
  $('#validation').innerHTML = '';
  $('#schema').textContent = '';
  $('#code').textContent = '';
  const manifest = $('#manifest');
  manifest.textContent = '';
  manifest.dataset.copy = '';
}

async function analyze() {
  hideError();
  resetOutputs();

  const dsl = $('#dsl').value.trim();
  const domain = $('#domain').value;
  const resultUnit = $('#resultUnit').value.trim() || null;
  const units = parseKV($('#units').value);
  const constraints = safeJSON($('#constraints').value);
  const pdeSpaceVars = ($('#pdeSpace').value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const pdeTimeVar = ($('#pdeTime').value || '').trim() || null;

  if (!dsl) {
    showError('Please enter an equation or DSL body.');
    return;
  }

  try {
    const translate = await api.post('/v1/translate', { dsl, domain });
    renderTranslate(translate);

    const validation = await api.post('/v1/validate', {
      dsl,
      domain,
      units,
      result_unit: resultUnit,
      constraints,
      pde_space_vars: pdeSpaceVars,
      pde_time_var: pdeTimeVar,
      checks: ['dimensional', 'constraints', 'relativistic', 'pde_class']
    });
    renderValidation(validation.report || validation, {
      units,
      resultUnit,
      domain
    });

    const schema = await api.post('/v1/schema', {
      name: 'equation',
      domain,
      dsl,
      units,
      constraints
    });
    const schemaPayload = schema.schema_json || schema.schema || schema;
    $('#schema').textContent = JSON.stringify(schemaPayload, null, 2);

    const code = await api.post('/v1/codegen', {
      dsl,
      name: 'compute',
      metadata: { doc: 'UI generated' }
    });
    $('#code').textContent = code.code || '';

    const manifestMeta = await api.post('/v1/manifest', {
      dsl,
      domain,
      units,
      constraints,
      result_unit: resultUnit,
      checks: ['dimensional', 'constraints', 'relativistic', 'pde_class'],
      pde_space_vars: pdeSpaceVars,
      pde_time_var: pdeTimeVar,
      metadata: { name: 'compute' }
    });

    let manifestFull = null;
    try {
      manifestFull = await api.get(`/v1/manifest/${manifestMeta.manifest_hash}`);
    } catch (err) {
      manifestFull = null;
    }
    renderManifest(manifestMeta, manifestFull);

    await renderInput();
    await renderCanonicalMath(translate.canonical);
  } catch (err) {
    showError(err.message || String(err));
    throw err;
  }
}

function renderTranslate(data) {
  $('#parsed').textContent = JSON.stringify(
    {
      success: data.success,
      expression: data.expression,
      canonical: data.canonical
    },
    null,
    2
  );
  const canonical = $('#canonicalText');
  canonical.dataset.copy = data.canonical || '';
  canonical.innerHTML = '';
  const raw = document.createElement('div');
  raw.className = 'canonical-raw';
  raw.textContent = data.canonical || '';
  canonical.appendChild(raw);
}

function renderValidation(report, context = {}) {
  const box = $('#validation');
  box.innerHTML = '';
  if (!report) return;

  const summary = document.createElement('div');
  summary.className = 'validation-item';
  const ok = report.ok === undefined ? true : report.ok;
  summary.innerHTML = `
    <h4>Status</h4>
    <div class="status"><span class="badge ${ok ? 'ok' : 'err'}">${ok ? 'ok' : 'failed'}</span></div>
  `;
  if (context.domain) {
    const domainLabel = document.createElement('div');
    domainLabel.className = 'domain-label';
    domainLabel.textContent = `domain • ${context.domain}`;
    summary.appendChild(domainLabel);
  }
  if (Array.isArray(report.errors) && report.errors.length) {
    const pre = document.createElement('pre');
    pre.className = 'codeblock';
    pre.textContent = JSON.stringify(report.errors, null, 2);
    summary.appendChild(pre);
  }
  box.appendChild(summary);

  const details = report.details || {};
  Object.entries(details).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'validation-item';
    const title = key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
    const flagOk = typeof value === 'object' && value !== null ? value.ok !== false : true;
    item.innerHTML = `
      <h4>${title}</h4>
      <div class="status"><span class="badge ${flagOk ? 'ok' : 'err'}">${flagOk ? 'ok' : 'issue'}</span></div>
    `;
    const extra = document.createElement('div');
    extra.className = 'mono';
    extra.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    item.appendChild(extra);
    box.appendChild(item);
  });

  if (context.units && Object.keys(context.units).length) {
    const unitsCard = document.createElement('div');
    unitsCard.className = 'validation-item';
    unitsCard.innerHTML = '<h4>Units</h4>';
    const chips = document.createElement('div');
    chips.className = 'unit-chips';
    Object.entries(context.units).forEach(([symbol, unit]) => {
      const chip = document.createElement('span');
      chip.className = 'unit-chip';
      chip.textContent = `${symbol} · ${unit}`;
      chips.appendChild(chip);
    });
    unitsCard.appendChild(chips);
    box.appendChild(unitsCard);
  }

  if (context.resultUnit) {
    const resultCard = document.createElement('div');
    resultCard.className = 'validation-item';
    resultCard.innerHTML = '<h4>Result Unit</h4>';
    const badge = document.createElement('span');
    badge.className = 'unit-chip highlight';
    badge.textContent = context.resultUnit;
    resultCard.appendChild(badge);
    box.appendChild(resultCard);
  }
}

function renderManifest(meta, manifest) {
  const manifestBox = $('#manifest');
  const payload = {
    manifest_hash: meta.manifest_hash,
    code_digest: meta.code_digest,
    manifest
  };
  const text = JSON.stringify(payload, null, 2);
  manifestBox.textContent = text;
  manifestBox.dataset.copy = text;
}

function showError(text) {
  $('#errorText').textContent = text;
  $('#errorCard').classList.remove('hidden');
}

function hideError() {
  $('#errorCard').classList.add('hidden');
}

function escapeHTML(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function typeset(el) {
  if (!window.MathJax || !el) return;
  await MathJax.typesetPromise([el]);
}

async function renderInput() {
  const raw = $('#dsl').value.trim();
  const mj = $('#rendered');
  mj.innerHTML = '';
  if (!raw) return;
  const pretty = raw.replace(/\*\*/g, '^');
  const span = document.createElement('span');
  span.innerHTML = `\\(${escapeHTML(pretty)}\\)`;
  mj.appendChild(span);
  await typeset(mj);
}

async function renderCanonicalMath(canonicalStr) {
  const canonical = $('#canonicalText');
  if (!canonicalStr) return;
  const math = document.createElement('div');
  math.className = 'canonical-math';
  math.innerHTML = `\\(${escapeHTML(canonicalStr.replace(/\*\*/g, '^'))}\\)`;
  canonical.appendChild(math);
  await typeset(math);
}

function copyText(selector) {
  const el = $(selector);
  if (!el) return;
  const text = el.dataset.copy || el.textContent || '';
  if (!text) return;
  navigator.clipboard.writeText(text);
}

function wire() {
  $('#analyzeBtn').addEventListener('click', async () => {
    try {
      await analyze();
    } catch (err) {
      console.error(err);
    }
  });
  $('#clearBtn').addEventListener('click', () => {
    $('#dsl').value = '';
    $('#units').value = '';
    $('#constraints').value = '';
    $('#parsed').textContent = '';
    $('#canonicalText').innerHTML = '';
    $('#canonicalText').dataset.copy = '';
    $('#schema').textContent = '';
    $('#code').textContent = '';
    $('#manifest').textContent = '';
    $('#manifest').dataset.copy = '';
    $('#validation').innerHTML = '';
    $('#pdeSpace').value = '';
    $('#pdeTime').value = '';
    $('#resultUnit').value = '';
    hideError();
    renderInput();
  });
  $$('.chip').forEach((btn) => {
    btn.addEventListener('click', () => setExample(btn.dataset.example));
  });
  $('#copyCanonical').addEventListener('click', () => copyText('#canonicalText'));
  $('#copyManifest').addEventListener('click', () => copyText('#manifest'));
  $('#dsl').addEventListener('input', () => {
    renderInput();
    hideError();
  });

  setExample('einstein');
  renderInput();
}

document.addEventListener('DOMContentLoaded', wire);
