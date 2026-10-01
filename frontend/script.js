const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

// ---- Prediction form -> backend /api/predict
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  Object.keys(data).forEach((k) => (data[k] = Number(data[k])));

  const box = $("result");
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Invalid input");
    const out = await res.json();
    box.className = "result " + (out.purchase ? "yes" : "no");
    box.textContent = `${out.label} (${Math.round(out.probability * 100)}% probability)`;
  } catch (err) {
    box.className = "result no";
    box.textContent = "Error: " + err.message;
  }
});

// ---- Model metrics -> /api/metrics
async function loadMetrics() {
  const m = await (await fetch("/api/metrics")).json();
  const pct = (v) => (v * 100).toFixed(1) + "%";
  $("metrics").innerHTML = ["accuracy", "precision", "recall", "f1"]
    .map((k) => `<div class="metric"><b>${pct(m[k])}</b>${pretty(k)}</div>`)
    .join("");
}

// ---- Feature importance chart -> /api/insights
async function loadInsights() {
  const d = await (await fetch("/api/insights")).json();
  const rows = Object.entries(d.correlation).sort((a, b) => b[1] - a[1]);
  $("chart").innerHTML = rows
    .map(([name, v]) => {
      const width = Math.min(Math.abs(v) * 100, 100);
      const color = v < 0 ? "var(--bad)" : "var(--primary)";
      return `<div class="bar-row">
        <span class="bar-label">${pretty(name)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${color}"></div></div>
        <span class="bar-val">${v}</span></div>`;
    })
    .join("");
}

loadMetrics();
loadInsights();
