import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function ProbabilityBar({ label, value, tone }) {
  return (
    <div className="probability-row">
      <div className="probability-label">{label}</div>
      <div className="probability-track">
        <div
          className={`probability-fill ${tone}`}
          style={{ width: `${(value * 100).toFixed(1)}%` }}
        />
      </div>
      <div className="probability-value">{value.toFixed(3)}</div>
    </div>
  );
}

function ModelCard({ title, row, prefix, chosenAlpha }) {
  if (!row) return null;
  return (
    <div className="model-card">
      <div className="model-title">{title}</div>
      <div className="chip-row">
        <span className="chip">Predicted: {row[`${prefix}_predicted_class`]}</span>
        <span className="chip">Set: {row[`${prefix}_prediction_set`]}</span>
        <span className="chip">alpha≈{chosenAlpha.toFixed(2)}</span>
      </div>
      <ProbabilityBar label="BLL" value={row[`${prefix}_bll`]} tone="red" />
      <ProbabilityBar label="FSRQ" value={row[`${prefix}_fsrq`]} tone="blue" />
    </div>
  );
}

function SummaryCard({ label, value }) {
  return (
    <div className="summary-card">
      <div className="summary-label">{label}</div>
      <div className="summary-value">{value}</div>
    </div>
  );
}

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [manualValues, setManualValues] = useState({});
  const [alpha, setAlpha] = useState(0.1);
  const [activeTab, setActiveTab] = useState("manual");
  const [inputMode, setInputMode] = useState("auto");
  const [selectedFile, setSelectedFile] = useState(null);
  const [manualResult, setManualResult] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/metadata`)
      .then((response) => response.json())
      .then((payload) => {
        setMetadata(payload);
        const defaults = {};
        for (const [key, meta] of Object.entries(payload.slider_features)) {
          defaults[key] = meta.transformed_default;
        }
        setManualValues(defaults);
      })
      .catch((err) => setError(err.message));
  }, []);

  const featureOrder = useMemo(
    () => metadata?.feature_columns?.join(", ") ?? "",
    [metadata],
  );
  const uploadPreviewColumns = useMemo(() => {
    if (!uploadResult?.rows?.length) return [];
    const available = Object.keys(uploadResult.rows[0]);
    const preferred = [
      "row_id",
      "Source_Name",
      "source_name",
      "bias_init_bll",
      "bias_init_fsrq",
      "bias_init_prediction_set",
      "greedy_supervised_bll",
      "greedy_supervised_fsrq",
      "greedy_supervised_prediction_set",
    ];
    const chosen = preferred.filter((column) => available.includes(column));
    return chosen.length ? chosen : available.slice(0, 8);
  }, [uploadResult]);

  async function handleManualPredict() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/predict/manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alpha, features: manualValues }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Manual prediction failed.");
      setManualResult(payload);
      setActiveTab("manual");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleUploadPredict() {
    if (!selectedFile) return;
    setBusy(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("alpha", String(alpha));
      formData.append("input_mode", inputMode);
      const response = await fetch(`${API_BASE}/predict/upload`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Upload prediction failed.");
      setUploadResult(payload);
      setActiveTab("upload");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function downloadUploadResults() {
    if (!uploadResult?.rows?.length) return;
    const columns = Object.keys(uploadResult.rows[0]);
    const lines = [
      columns.join(","),
      ...uploadResult.rows.map((row) =>
        columns
          .map((column) => JSON.stringify(row[column] ?? ""))
          .join(","),
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "blazar_classification_predictions.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  if (!metadata) {
    return <div className="app-shell"><div className="loading-card">Loading Blazar Classification…</div></div>;
  }

  const manualRow = manualResult?.rows?.[0];
  const uploadRows = uploadResult?.rows?.slice(0, 20) ?? [];

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>Blazar Classification</h1>
          <p>Classifier for Blazar Classes of Unknown type using Bias Initialization and Greedy Supervised models.</p>
        </div>
        <div className="hero-badges">
          <span>Dual-model inference</span>
          <span>Automatic log handling</span>
          <span>Conformal prediction sets</span>
          <span>Manual + batch input</span>
        </div>
      </header>

      <div className="global-bar">
        <div className="feature-order">Feature order: {featureOrder}</div>
        <label className="alpha-control">
          <span>Conformal alpha</span>
          <input
            type="range"
            min="0.01"
            max="0.50"
            step="0.01"
            value={alpha}
            onChange={(event) => setAlpha(Number(event.target.value))}
          />
          <strong>{alpha.toFixed(2)}</strong>
        </label>
      </div>

      <main className="workspace">
        <section className="panel controls-panel">
          <div className="panel-header">
            <h2>Input</h2>
            <p>Use compact transformed sliders for one source, or upload a .csv / .npy batch.</p>
          </div>

          <div className="tab-row">
            <button className={activeTab === "manual" ? "active" : ""} onClick={() => setActiveTab("manual")}>
              Manual source
            </button>
            <button className={activeTab === "upload" ? "active" : ""} onClick={() => setActiveTab("upload")}>
              Batch upload
            </button>
          </div>

          {activeTab === "manual" && (
            <>
              <div className="slider-grid">
                {metadata.feature_columns.map((feature) => {
                  const meta = metadata.slider_features[feature];
                  const value = manualValues[feature] ?? meta.transformed_default;
                  return (
                    <label key={feature} className="slider-card">
                      <div className="slider-head">
                        <span>{feature}</span>
                        <small>{meta.logged ? "log-aware" : "direct scale"}</small>
                      </div>
                      <input
                        type="range"
                        min={meta.transformed_min}
                        max={meta.transformed_max}
                        step={(meta.transformed_max - meta.transformed_min) / 150}
                        value={value}
                        onChange={(event) =>
                          setManualValues((current) => ({
                            ...current,
                            [feature]: Number(event.target.value),
                          }))
                        }
                      />
                      <div className="slider-meta">
                        <span>{value.toFixed(3)}</span>
                        <span>
                          raw default {meta.raw_default >= 0.001 ? meta.raw_default.toPrecision(4) : meta.raw_default.toExponential(2)}
                        </span>
                      </div>
                    </label>
                  );
                })}
              </div>
              <button className="primary-button" onClick={handleManualPredict} disabled={busy}>
                {busy ? "Predicting…" : "Predict"}
              </button>
            </>
          )}

          {activeTab === "upload" && (
            <div className="upload-panel">
              <label className="upload-dropzone">
                <input
                  type="file"
                  accept=".csv,.npy"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
                <div>{selectedFile ? selectedFile.name : "Choose a .csv or .npy file"}</div>
                <small>The first 7 columns can be used if the feature names are not present.</small>
              </label>
              <div className="input-mode-row">
                {metadata.input_modes.map((mode) => (
                  <button
                    key={mode.value}
                    className={inputMode === mode.value ? "mode-pill active" : "mode-pill"}
                    onClick={() => setInputMode(mode.value)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
              <button className="primary-button" onClick={handleUploadPredict} disabled={busy || !selectedFile}>
                {busy ? "Scoring…" : "Score uploaded batch"}
              </button>
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}
        </section>

        <section className="panel results-panel">
          <div className="panel-header">
            <h2>Results</h2>
            <p>Bias Initialization and Greedy Supervised predictions stay aligned through one shared inference core.</p>
          </div>

          <div className="set-guide-grid">
            <div className="set-guide-card">
              <div className="set-guide-title">Set = BLL</div>
              <div className="set-guide-copy">Only BLL passes the conformal threshold. This is a confident single-label result.</div>
            </div>
            <div className="set-guide-card">
              <div className="set-guide-title">Set = FSRQ</div>
              <div className="set-guide-copy">Only FSRQ passes the conformal threshold. This is a confident single-label result.</div>
            </div>
            <div className="set-guide-card">
              <div className="set-guide-title">Set = BLL, FSRQ</div>
              <div className="set-guide-copy">Both labels pass. The sample is ambiguous at the current alpha.</div>
            </div>
            <div className="set-guide-card">
              <div className="set-guide-title">Set = Empty</div>
              <div className="set-guide-copy">Neither label passes. This usually signals an uncertain or out-of-distribution sample.</div>
            </div>
          </div>

          {manualRow && activeTab === "manual" && (
            <div className="manual-results-grid">
              <ModelCard
                title="Bias Initialization"
                row={manualRow}
                prefix="bias_init"
                chosenAlpha={manualResult.chosen_alphas.bias_init}
              />
              <ModelCard
                title="Greedy Supervised"
                row={manualRow}
                prefix="greedy_supervised"
                chosenAlpha={manualResult.chosen_alphas.greedy_supervised}
              />
            </div>
          )}

          {uploadResult && activeTab === "upload" && (
            <>
              <div className="upload-summary-note">
                Detected upload mode: <strong>{uploadResult.detected_input_mode}</strong>
              </div>
              <div className="upload-summary-grid">
                <div className="upload-summary-group">
                  <h3>Bias Initialization</h3>
                  <div className="summary-grid">
                    <SummaryCard label="BLL predictions" value={uploadResult.batch_summary.bias_init.bll_predictions} />
                    <SummaryCard label="FSRQ predictions" value={uploadResult.batch_summary.bias_init.fsrq_predictions} />
                    <SummaryCard label="Singleton sets" value={uploadResult.batch_summary.bias_init.bll_only_sets + uploadResult.batch_summary.bias_init.fsrq_only_sets} />
                    <SummaryCard label="Ambiguous / empty" value={uploadResult.batch_summary.bias_init.both_sets + uploadResult.batch_summary.bias_init.empty_sets} />
                  </div>
                </div>
                <div className="upload-summary-group">
                  <h3>Greedy Supervised</h3>
                  <div className="summary-grid">
                    <SummaryCard label="BLL predictions" value={uploadResult.batch_summary.greedy_supervised.bll_predictions} />
                    <SummaryCard label="FSRQ predictions" value={uploadResult.batch_summary.greedy_supervised.fsrq_predictions} />
                    <SummaryCard label="Singleton sets" value={uploadResult.batch_summary.greedy_supervised.bll_only_sets + uploadResult.batch_summary.greedy_supervised.fsrq_only_sets} />
                    <SummaryCard label="Ambiguous / empty" value={uploadResult.batch_summary.greedy_supervised.both_sets + uploadResult.batch_summary.greedy_supervised.empty_sets} />
                  </div>
                </div>
              </div>

              <div className="table-actions">
                <button className="ghost-button" onClick={downloadUploadResults}>Download combined predictions</button>
                <span>Previewing {uploadRows.length} rows</span>
              </div>

              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      {uploadRows[0] && uploadPreviewColumns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {uploadRows.map((row, index) => (
                      <tr key={index}>
                        {uploadPreviewColumns.map((column) => (
                          <td key={column}>{String(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {!manualRow && !uploadResult && (
            <div className="empty-state">
              Run a manual prediction or upload a batch to see probabilities and conformal sets for both models.
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
