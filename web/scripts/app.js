const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const api = {
  async request(method, path, body) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (err) {
      payload = text;
    }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : payload;
      const message = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
      throw new Error(message || response.statusText);
    }
    return payload;
  },
  get(path) {
    return api.request('GET', path);
  },
  post(path, body) {
    return api.request('POST', path, body);
  }
};

const KEYWORDS = new Set([
  'return', 'sin', 'cos', 'tan', 'exp', 'sqrt', 'log', 'Eq', 'd', 'd2', 'diff', 'gamma'
]);

let liveValidationTimer = null;
let detectedVariables = [];

function parseUnitsText(text) {
  const units = {};
  const warnings = [];
  if (!text) return { units, warnings };

  text.split(/\r?\n/).forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) return;
    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) {
      warnings.push(`Line ${index + 1}: missing ':'`);
      return;
    }
    const name = line.slice(0, colonIndex).trim();
    let unit = line.slice(colonIndex + 1).split('#')[0].trim();
    unit = unit.replace(/[\s,;]+$/, '');
    if (!name) {
      warnings.push(`Line ${index + 1}: empty name`);
      return;
    }
    if (!unit) {
      warnings.push(`Line ${index + 1}: empty unit for ${name}`);
      return;
    }
    units[name] = unit;
  });

  return { units, warnings };
}

function unitsMapToText(map) {
  return Object.entries(map)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n');
}

function detectVariables(dsl) {
  if (!dsl) return [];
  const tokens = dsl.match(/[A-Za-z][A-Za-z0-9_]*/g) || [];
  const seen = new Set();
  const variables = [];
  tokens.forEach((token) => {
    if (KEYWORDS.has(token)) return;
    if (!seen.has(token)) {
      seen.add(token);
      variables.push(token);
    }
  });
  return variables.sort();
}

function renderDetectedVariables() {
  const container = $('#detected-vars');
  if (!container) return;
  const { units } = parseUnitsText($('#units').value);
  container.innerHTML = '';
  detectedVariables.forEach((name) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const unit = units[name];
    chip.textContent = unit ? `${name} · ${unit}` : name;
    container.appendChild(chip);
  });
}

async function autoFillUnits() {
  if (detectedVariables.length === 0) return;
  const { units } = parseUnitsText($('#units').value);
  const system = $('#unit-system').value || 'SI';
  try {
    const suggestions = await api.post('/v1/units/suggest', {
      variables: detectedVariables,
      system,
      existing: units
    });
    const merged = { ...units, ...suggestions.suggestions };
    $('#units').value = unitsMapToText(merged);
    renderDetectedVariables();
    scheduleLiveValidation();
  } catch (err) {
    showError(err.message || String(err));
  }
}

async function inferUnits() {
  const dsl = $('#dsl').value;
  if (!dsl.trim()) return;
  const parsed = parseUnitsText($('#units').value);
  try {
    const response = await api.post('/v1/units/infer', {
      dsl,
      known: parsed.units
    });
    const merged = { ...parsed.units, ...response.units };
    $('#units').value = unitsMapToText(merged);
    renderDetectedVariables();
    scheduleLiveValidation(response.trace);
  } catch (err) {
    showError(err.message || String(err));
  }
}

function scheduleLiveValidation(trace) {
  if (liveValidationTimer) {
    clearTimeout(liveValidationTimer);
  }
  liveValidationTimer = setTimeout(() => liveValidate(trace), 250);
}

async function liveValidate(trace) {
  const statusBox = $('#live-validation');
  const parsed = parseUnitsText($('#units').value);
  try {
    const response = await api.post('/v1/units/validate', { units: parsed.units });
    if (response.ok) {
      statusBox.textContent = '✓ Units look valid';
      statusBox.style.color = 'var(--ok)';
    } else {
      const messages = response.diagnostics.map((diag) => `${diag.symbol}: ${diag.message}`);
      statusBox.textContent = `✗ ${messages.join(' | ')}`;
      statusBox.style.color = 'var(--err)';
    }
    if (trace) {
      statusBox.textContent += ` • ${trace.map((step) => step.detail).join(' › ')}`;
    }
  } catch (err) {
    statusBox.textContent = `✗ ${err.message || String(err)}`;
    statusBox.style.color = 'var(--err)';
  }
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

function updateDetectedFromDSL() {
  detectedVariables = detectVariables($('#dsl').value);
  renderDetectedVariables();
}

async function analyze() {
  hideError();
  resetOutputs();

  const dsl = $('#dsl').value.trim();
  if (!dsl) {
    showError('Please enter an equation or DSL body.');
    return;
  }

  const domain = $('#domain').value;
  const resultUnit = $('#resultUnit').value.trim() || null;
  const unitsText = $('#units').value;
  const parsedUnits = parseUnitsText(unitsText);
  const constraints = safeJSON($('#constraints').value);
  const pdeSpaceVars = ($('#pdeSpace').value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const pdeTimeVar = ($('#pdeTime').value || '').trim() || null;

  try {
    const translate = await api.post('/v1/translate', { dsl, domain });
    renderTranslate(translate);

    const validation = await api.post('/v1/validate', {
      dsl,
      domain,
      units: parsedUnits.units,
      result_unit: resultUnit,
      constraints,
      pde_space_vars: pdeSpaceVars,
      pde_time_var: pdeTimeVar,
      checks: ['dimensional', 'constraints', 'relativistic', 'pde_class']
    });
    renderValidation(validation.report || validation);

    const schema = await api.post('/v1/schema', {
      name: 'equation',
      domain,
      dsl,
      units: parsedUnits.units,
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
      units: parsedUnits.units,
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
}

function renderValidation(report) {
  const container = $('#validation');
  container.innerHTML = '';

  const unitBlock = document.createElement('div');
  unitBlock.className = 'validation-block';
  const unitOk = !report.unit_diagnostics || report.unit_diagnostics.length === 0;
  unitBlock.innerHTML = `<h4>Units ${unitOk ? '✓' : '✗'}</h4>`;
  if (report.units) {
    const list = document.createElement('pre');
    list.textContent = JSON.stringify(report.units, null, 2);
    unitBlock.appendChild(list);
  }
  if (!unitOk) {
    const diagList = document.createElement('ul');
    report.unit_diagnostics.forEach((diag) => {
      const item = document.createElement('li');
      item.textContent = `${diag.symbol}: ${diag.message}`;
      diagList.appendChild(item);
    });
    unitBlock.appendChild(diagList);
  }
  container.appendChild(unitBlock);

  const dimBlock = document.createElement('div');
  dimBlock.className = 'validation-block';
  const dimOk = report.dimensions?.ok !== false;
  dimBlock.innerHTML = `<h4>Dimensions ${dimOk ? '✓' : '✗'}</h4>`;
  if (report.dimensions?.summary?.length) {
    const list = document.createElement('ul');
    report.dimensions.summary.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = `${item.equation}: ${item.lhs}`;
      if (item.rhs && item.rhs !== item.lhs) {
        li.textContent += ` vs ${item.rhs}`;
      }
      list.appendChild(li);
    });
    dimBlock.appendChild(list);
  }
  if (report.dimensions?.error) {
    const error = document.createElement('div');
    error.className = 'error-text';
    error.textContent = report.dimensions.error;
    dimBlock.appendChild(error);
  }
  container.appendChild(dimBlock);

  const pipelineBlock = document.createElement('div');
  pipelineBlock.className = 'validation-block';
  const pipelineOk = report.pipeline?.ok !== false;
  pipelineBlock.innerHTML = `<h4>Pipeline ${pipelineOk ? '✓' : '✗'}</h4>`;
  if (report.pipeline?.errors?.length) {
    const list = document.createElement('ul');
    report.pipeline.errors.forEach((err) => {
      const item = document.createElement('li');
      item.textContent = `${err.stage || 'pipeline'}: ${err.message}`;
      list.appendChild(item);
    });
    pipelineBlock.appendChild(list);
  }
  container.appendChild(pipelineBlock);
}

function renderManifest(meta, full) {
  const manifest = $('#manifest');
  const payload = { meta, full };
  manifest.textContent = JSON.stringify(payload, null, 2);
  manifest.dataset.copy = manifest.textContent;
}

function showError(message) {
  const card = $('#errorCard');
  const text = $('#errorText');
  text.textContent = message;
  card.classList.remove('hidden');
}

function hideError() {
  const card = $('#errorCard');
  if (!card.classList.contains('hidden')) {
    card.classList.add('hidden');
  }
}

function attachEventListeners() {
  $('#analyzeBtn').addEventListener('click', analyze);
  $('#clearBtn').addEventListener('click', () => {
    ['dsl', 'units', 'resultUnit', 'constraints', 'pdeSpace', 'pdeTime'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    detectedVariables = [];
    renderDetectedVariables();
    scheduleLiveValidation();
    resetOutputs();
  });

  $('#autoFillUnits').addEventListener('click', autoFillUnits);
  $('#inferUnits').addEventListener('click', inferUnits);
  $('#unit-system').addEventListener('change', autoFillUnits);

  $('#units').addEventListener('input', () => {
    renderDetectedVariables();
    scheduleLiveValidation();
  });

  $('#dsl').addEventListener('input', () => {
    updateDetectedFromDSL();
    scheduleLiveValidation();
  });

  const copyCanonical = $('#copyCanonical');
  if (copyCanonical) {
    copyCanonical.addEventListener('click', () => copyText('#canonicalText'));
  }
  const copyManifest = $('#copyManifest');
  if (copyManifest) {
    copyManifest.addEventListener('click', () => copyText('#manifest'));
  }
}

function initExamples() {
  $$('.chip[data-example]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const kind = chip.dataset.example;
      setExample(kind);
      updateDetectedFromDSL();
      scheduleLiveValidation();
    });
  });
}

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
  renderDetectedVariables();
}

async function renderInput() {
  const dslText = $('#dsl').value;
  if (!dslText.trim()) {
    $('#rendered').textContent = '';
    return;
  }

  try {
    const translate = await api.post('/v1/translate', { dsl: dslText, domain: $('#domain').value });
    $('#rendered').textContent = translate.canonical;
  } catch (err) {
    $('#rendered').textContent = '';
  }
}

async function renderCanonicalMath(canonical) {
  const box = $('#canonicalText');
  box.textContent = canonical;
  box.dataset.copy = canonical;
  if (window.MathJax) {
    window.MathJax.typesetPromise([box]).catch(() => {});
  }
}

function copyText(selector) {
  const el = $(selector);
  if (!el) return;
  const text = el.dataset.copy || el.textContent || '';
  if (!text) return;
  navigator.clipboard.writeText(text).catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  attachEventListeners();
  initExamples();
  setExample('einstein');
  hideError();
  updateDetectedFromDSL();
  scheduleLiveValidation();
});

