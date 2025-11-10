async function fetchJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

function renderSummary(summary) {
  document.querySelector("#statute-count").textContent = summary.sections_total ?? 0;
  document.querySelector("#pair-count").textContent = summary.pairs_total ?? 0;
  const list = document.querySelector("#top-sections");
  list.innerHTML = "";
  (summary.top_sections || []).forEach((section) => {
    const li = document.createElement("li");
    li.textContent = `${section.identifier || section.section || "Unknown"} — F_policy ${(section.policy_flaw ?? 0).toFixed(3)}`;
    list.appendChild(li);
  });
  const timeseries = document.querySelector("#timeseries");
  if (summary.time_series && Object.keys(summary.time_series).length > 0) {
    timeseries.textContent = JSON.stringify(summary.time_series, null, 2);
  }
}

async function boot() {
  try {
    const summary = await fetchJSON("/api/law/tax/summary");
    renderSummary(summary);
    const heatmap = document.querySelector("#heatmap");
    const pairs = await fetchJSON("/api/law/tax/pairs?limit=10");
    heatmap.textContent = pairs.results.length > 0 ? JSON.stringify(pairs.results.slice(0, 3), null, 2) : "No scored pairs yet.";
    const drilldown = document.querySelector("#drilldown");
    drilldown.textContent = "Select a section via future interactions.";
  } catch (error) {
    const heatmap = document.querySelector("#heatmap");
    heatmap.textContent = error instanceof Error ? error.message : "Failed to load dashboard.";
  }
}

document.addEventListener("DOMContentLoaded", boot);
