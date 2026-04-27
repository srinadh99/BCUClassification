from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blazar_classification_app.inference import FEATURE_COLUMNS, get_service


st.set_page_config(
    page_title="Blazar Classification",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 96rem;}
    .stApp {background: radial-gradient(circle at top left, #f7fbff 0%, #eef4f7 42%, #f8fafb 100%);}
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #0ea5e9 100%);
        color: white;
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 0.35rem 0; font-size: 2rem; line-height: 1.1;}
    .hero p {margin: 0; opacity: 0.92; font-size: 0.98rem;}
    .hero-badges {display: flex; gap: 0.5rem; margin-top: 0.85rem; flex-wrap: wrap;}
    .hero-badge {
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.8rem;
    }
    .panel {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 0.9rem 1rem 1rem 1rem;
        box-shadow: 0 18px 32px rgba(15, 23, 42, 0.06);
    }
    .compact-note {
        font-size: 0.85rem;
        color: #475569;
        margin: 0.2rem 0 0.8rem 0;
    }
    .result-card {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.95));
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 0.9rem;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.55rem;
    }
    .chip-row {display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.65rem;}
    .chip {
        font-size: 0.8rem;
        padding: 0.32rem 0.6rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    .prob-row {display: grid; grid-template-columns: 74px 1fr 54px; align-items: center; gap: 0.55rem; margin-bottom: 0.55rem;}
    .prob-label {font-size: 0.86rem; color: #334155; font-weight: 600;}
    .prob-track {
        height: 11px;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
    }
    .prob-fill-red {
        height: 100%;
        background: linear-gradient(90deg, #fb7185, #ef4444);
    }
    .prob-fill-blue {
        height: 100%;
        background: linear-gradient(90deg, #60a5fa, #2563eb);
    }
    .prob-value {font-size: 0.82rem; color: #0f172a; font-weight: 700; text-align: right;}
    .summary-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.65rem;}
    .summary-box {
        padding: 0.65rem 0.75rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid rgba(148,163,184,0.22);
        min-height: 82px;
    }
    .summary-label {font-size: 0.74rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;}
    .summary-value {font-size: 1.15rem; color: #0f172a; font-weight: 800; margin-top: 0.18rem;}
    .feature-order {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem;
        color: #334155;
        background: rgba(255,255,255,0.85);
        padding: 0.55rem 0.75rem;
        border-radius: 14px;
        border: 1px dashed rgba(148,163,184,0.45);
        margin: 0.75rem 0 0.2rem 0;
    }
    .set-guide {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.35rem;
        margin-bottom: 0.3rem;
    }
    .set-guide-card {
        padding: 0.75rem 0.85rem;
        border-radius: 16px;
        background: rgba(248,250,252,0.96);
        border: 1px solid rgba(148,163,184,0.18);
    }
    .set-guide-title {
        font-size: 0.82rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.18rem;
    }
    .set-guide-copy {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


service = get_service()
metadata = service.metadata_payload()

st.markdown(
    f"""
    <div class="hero">
        <h1>Blazar Classification</h1>
        <p>Classifier for Blazar Classes of Unknown type using Bias Initialization and Greedy Supervised models.</p>
        <div class="hero-badges">
            <span class="hero-badge">Dual-model inference</span>
            <span class="hero-badge">Automatic log handling for uploads</span>
            <span class="hero-badge">Conformal prediction sets</span>
            <span class="hero-badge">Manual + batch input</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

alpha = st.slider("Conformal alpha", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
st.markdown(
    f'<div class="feature-order">Feature order: {", ".join(FEATURE_COLUMNS)}</div>',
    unsafe_allow_html=True,
)

controls_col, results_col = st.columns([1.1, 1.0], gap="large")

with controls_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Input")
    st.markdown(
        '<p class="compact-note">Use compact transformed sliders for one source, or upload a .csv / .npy batch. Uploads can be raw or already log-transformed.</p>',
        unsafe_allow_html=True,
    )
    tab_manual, tab_upload = st.tabs(["Manual source", "Batch upload"])

    manual_prediction = None
    upload_prediction = None

    with tab_manual:
        slider_columns = st.columns(2, gap="medium")
        manual_values = {}
        for index, feature_name in enumerate(FEATURE_COLUMNS):
            feature_meta = metadata["slider_features"][feature_name]
            column = slider_columns[index % 2]
            with column:
                manual_values[feature_name] = st.slider(
                    feature_name,
                    min_value=float(feature_meta["transformed_min"]),
                    max_value=float(feature_meta["transformed_max"]),
                    value=float(feature_meta["transformed_default"]),
                    help=(
                        "This slider uses the same transformed feature space as the notebooks."
                        if feature_meta["logged"]
                        else "This feature is not log-transformed in preprocessing."
                    ),
                )
                raw_preview = feature_meta["raw_default"]
                st.caption(
                    f"BCU default raw preview: {raw_preview:.4g}"
                    if raw_preview >= 0.001
                    else f"BCU default raw preview: {raw_preview:.3e}"
                )

        if st.button("Predict", type="primary", use_container_width=True):
            manual_prediction = service.predict(
                service.make_manual_feature_frame(manual_values),
                input_mode="transformed",
                alpha=alpha,
            )
            st.session_state["manual_prediction"] = manual_prediction

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload .csv or .npy",
            type=["csv", "npy"],
            accept_multiple_files=False,
            help="CSV files can include the seven model features directly or as the first seven columns.",
        )
        input_mode = st.radio(
            "Upload preprocessing mode",
            options=["auto", "raw", "transformed"],
            format_func=lambda value: {
                "auto": "Auto detect",
                "raw": "Raw features",
                "transformed": "Already log-transformed",
            }[value],
            horizontal=True,
        )
        if st.button("Score uploaded batch", use_container_width=True, disabled=uploaded_file is None):
            if uploaded_file is not None:
                upload_features, upload_extras = service.parse_uploaded_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                upload_prediction = service.predict(
                    upload_features,
                    input_mode=input_mode,
                    alpha=alpha,
                    extras=upload_extras,
                )
                st.session_state["upload_prediction"] = upload_prediction

    st.markdown("</div>", unsafe_allow_html=True)


def render_model_card(title: str, row: dict, prefix: str, chosen_alpha: float) -> None:
    bll_value = float(row[f"{prefix}_bll"])
    fsrq_value = float(row[f"{prefix}_fsrq"])
    conformal_set = row[f"{prefix}_prediction_set"]
    predicted_class = row[f"{prefix}_predicted_class"]
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{title}</div>
            <div class="chip-row">
                <span class="chip">Predicted: {predicted_class}</span>
                <span class="chip">Set: {conformal_set}</span>
                <span class="chip">alpha≈{chosen_alpha:.2f}</span>
            </div>
            <div class="prob-row">
                <div class="prob-label">BLL</div>
                <div class="prob-track"><div class="prob-fill-red" style="width:{bll_value * 100:.1f}%"></div></div>
                <div class="prob-value">{bll_value:.3f}</div>
            </div>
            <div class="prob-row">
                <div class="prob-label">FSRQ</div>
                <div class="prob-track"><div class="prob-fill-blue" style="width:{fsrq_value * 100:.1f}%"></div></div>
                <div class="prob-value">{fsrq_value:.3f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with results_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Results")
    st.markdown(
        """
        <div class="set-guide">
            <div class="set-guide-card">
                <div class="set-guide-title">Set = BLL</div>
                <div class="set-guide-copy">Only the BLL label passes the conformal threshold. This is a confident single-label result.</div>
            </div>
            <div class="set-guide-card">
                <div class="set-guide-title">Set = FSRQ</div>
                <div class="set-guide-copy">Only the FSRQ label passes the conformal threshold. This is a confident single-label result.</div>
            </div>
            <div class="set-guide-card">
                <div class="set-guide-title">Set = BLL, FSRQ</div>
                <div class="set-guide-copy">Both labels pass. The model is not separating the sample cleanly at this alpha, so treat it as ambiguous.</div>
            </div>
            <div class="set-guide-card">
                <div class="set-guide-title">Set = Empty</div>
                <div class="set-guide-copy">Neither label passes. This usually means the sample looks out-of-distribution or especially uncertain.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_manual = st.session_state.get("manual_prediction")
    active_upload = st.session_state.get("upload_prediction")

    if active_manual:
        row = active_manual["rows"][0]
        left_card, right_card = st.columns(2, gap="medium")
        with left_card:
            render_model_card(
                "Bias Initialization",
                row,
                "bias_init",
                active_manual["chosen_alphas"]["bias_init"],
            )
        with right_card:
            render_model_card(
                "Greedy Supervised",
                row,
                "greedy_supervised",
                active_manual["chosen_alphas"]["greedy_supervised"],
            )

    if active_upload:
        detected_mode = active_upload["detected_input_mode"]
        st.markdown(
            f'<p class="compact-note">Detected upload mode: <strong>{detected_mode}</strong>. Download the combined results below.</p>',
            unsafe_allow_html=True,
        )
        for model_key, model_title in [
            ("bias_init", "Bias Initialization"),
            ("greedy_supervised", "Greedy Supervised"),
        ]:
            summary = active_upload["batch_summary"][model_key]
            st.markdown(f"#### {model_title}")
            st.markdown(
                f"""
                <div class="summary-grid">
                    <div class="summary-box"><div class="summary-label">BLL predictions</div><div class="summary-value">{summary['bll_predictions']}</div></div>
                    <div class="summary-box"><div class="summary-label">FSRQ predictions</div><div class="summary-value">{summary['fsrq_predictions']}</div></div>
                    <div class="summary-box"><div class="summary-label">Singletons</div><div class="summary-value">{summary['bll_only_sets'] + summary['fsrq_only_sets']}</div></div>
                    <div class="summary-box"><div class="summary-label">Ambiguous / empty</div><div class="summary-value">{summary['both_sets'] + summary['empty_sets']}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        results_df = pd.DataFrame(active_upload["rows"])
        csv_payload = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download combined predictions",
            data=csv_payload,
            file_name="blazar_classification_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        preview_columns = [
            column
            for column in [
                "row_id",
                "Source_Name",
                "source_name",
                "bias_init_bll",
                "bias_init_fsrq",
                "bias_init_prediction_set",
                "greedy_supervised_bll",
                "greedy_supervised_fsrq",
                "greedy_supervised_prediction_set",
            ]
            if column in results_df.columns
        ]
        st.dataframe(
            results_df[preview_columns].head(25) if preview_columns else results_df.head(25),
            use_container_width=True,
            height=320,
        )

    if not active_manual and not active_upload:
        st.info("Run a manual prediction or score an uploaded file to see both model outputs here.")

    st.markdown("</div>", unsafe_allow_html=True)
