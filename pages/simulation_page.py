import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
TEST_CSV   = os.path.join(BASE_DIR, "data", "cc_fraud_test_processed.csv")

# 대표 시뮬레이션 모델. 전처리 Pipeline이 포함되어 있어 processed DataFrame 전체를 입력한다.
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgboost_tuned.pkl")


def _fmt_int(value):
    return f"{int(value):,}"


def _fmt_float(value, digits=4):
    return f"{float(value):.{digits}f}"


def _chart_layout(height=300):
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#071426",
        height=height,
        margin=dict(l=8, r=8, t=40, b=24),
        font=dict(color="#D6E4FF"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zeroline=False,
                   tickfont=dict(color="#93A4C7"), title_font=dict(color="#D6E4FF")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zeroline=False,
                   tickfont=dict(color="#93A4C7"), title_font=dict(color="#D6E4FF")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#D6E4FF", size=10)),
    )


def _render_kpi_card(label, value, color):
    return (
        f'<div class="sim-kpi-card">'
        f'<span style="color:#9FC4FF;font-size:12px;font-weight:700;display:block;margin-bottom:6px;">{label}</span>'
        f'<span style="color:{color};font-size:24px;font-weight:900;display:block;line-height:1;">{value}</span>'
        f'</div>'
    )


@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="테스트 데이터 로딩 & 예측 중...", ttl=3600)
def load_predictions():
    df = pd.read_csv(TEST_CSV)

    required_cols = [
        "merchant", "category", "amt", "gender", "state", "zip", "lat", "long",
        "city_pop", "unix_time", "merch_lat", "merch_long", "trans_hour",
        "trans_dayofweek", "trans_month", "trans_day", "age", "distance_km",
        "amt_zscore", "hour_dev", "high_amt_far", "is_fraud",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"시뮬레이션 입력 데이터에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}")

    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])

    model = load_model()
    if hasattr(model, "steps") and hasattr(model.steps[-1][1], "n_jobs"):
        model.steps[-1][1].n_jobs = 1

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raw_pred = np.asarray(model.predict(X))
        y_prob = raw_pred[:, 1] if raw_pred.ndim == 2 and raw_pred.shape[1] > 1 else raw_pred.astype(float)

    return y, y_prob, len(df), int(y.sum())


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    f2 = 5 * pr * rc / (4 * pr + rc) if (4 * pr + rc) > 0 else 0.0
    return {"precision": round(pr,4), "recall": round(rc,4),
            "f1": round(f1,4), "f2": round(f2,4), "tp": tp, "fp": fp, "fn": fn}


@st.cache_data(show_spinner=False, ttl=3600)
def build_threshold_table(_y_true, _y_prob):
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        m = compute_metrics_at_threshold(_y_true, _y_prob, t)
        m["threshold"] = round(float(t), 2)
        rows.append(m)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def global_aucs(_y_true, _y_prob):
    return round(roc_auc_score(_y_true, _y_prob), 4), round(average_precision_score(_y_true, _y_prob), 4)


def show_simulation_page():
    y_true, y_prob, total_tx, fraud_tx = load_predictions()
    df_table = build_threshold_table(y_true, y_prob)
    roc_auc, pr_auc = global_aucs(y_true, y_prob)
    best_row = df_table.loc[df_table["f2"].idxmax()]
    best_t   = float(best_row["threshold"])
    default_threshold = round(best_t, 2)

    if st.session_state.get("simulation_model_path") != MODEL_PATH:
        st.session_state["simulation_model_path"] = MODEL_PATH
        st.session_state["threshold_val"] = default_threshold
        st.session_state["threshold_slider_widget"] = default_threshold
    elif "threshold_slider_widget" not in st.session_state:
        st.session_state["threshold_slider_widget"] = st.session_state.get("threshold_val", default_threshold)

    st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: linear-gradient(180deg,rgba(9,20,40,0.97),rgba(6,15,30,0.99)) !important;
        border:1px solid rgba(120,163,230,0.18) !important;
        border-radius:16px !important;
        box-shadow:0 18px 42px rgba(1,6,18,0.30),inset 0 1px 0 rgba(255,255,255,0.04) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom:12px !important; }
    [data-testid="stSlider"] { padding-top:8px !important; padding-bottom:2px !important; }
    [data-testid="stPlotlyChart"] { margin-top:0 !important; margin-bottom:0 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
        background:rgba(66,111,189,0.18) !important; border:1px solid rgba(120,163,230,0.28) !important;
        color:#9FC4FF !important; font-size:12px !important; font-weight:700 !important;
        border-radius:8px !important; padding:4px 12px !important;
    }
    .sim-kpi-card { min-height:78px; padding:12px; border-radius:14px;
        background:linear-gradient(180deg,rgba(18,32,55,0.98),rgba(8,19,35,0.98));
        border:1px solid rgba(120,163,230,0.18); display:flex; flex-direction:column; justify-content:center; }
    .sim-kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }
    .sim-summary-grid { min-height:96px; display:grid; grid-template-columns:repeat(5,1fr); gap:0; align-items:stretch; }
    .cm-table { width:100%; border-collapse:collapse; font-size:13px; margin-bottom:10px; }
    .cm-table th { background:rgba(66,111,189,0.18); color:#A8C4E8; padding:9px 6px; text-align:center;
        border:1px solid rgba(120,163,230,0.18); font-weight:800; font-size:12px; }
    .cm-table td { padding:9px 6px; text-align:center; border:1px solid rgba(120,163,230,0.14); color:#D6E4FF; }
    .cm-table td.label { background:rgba(66,111,189,0.10); color:#93A8CC; font-weight:700;
        text-align:left; padding-left:10px; font-size:12px; }
    .cm-table td.tp  { background:rgba(20,83,45,0.35);  color:#4DE8A0; font-weight:900; font-size:14px; }
    .cm-table td.fp  { background:rgba(90,10,10,0.35);  color:#FF7A7A; font-weight:900; font-size:14px; }
    .cm-table td.fn  { background:rgba(80,30,7,0.35);   color:#FFB14E; font-weight:900; font-size:14px; }
    .cm-table td.tn  { background:rgba(15,23,42,0.35);  color:#AAB9D6; font-weight:900; font-size:14px; }
    .cm-table td.total { background:rgba(66,111,189,0.10); color:#9EBBFF; font-weight:700; }
    .metric-box { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .metric-item { background:rgba(66,111,189,0.10); border:1px solid rgba(120,163,230,0.14);
        border-radius:10px; padding:10px 8px; text-align:center; }
    .metric-label { color:#8FB0E0; font-size:11px; font-weight:700; margin-bottom:5px; }
    .metric-value-red  { color:#FF7A7A; font-size:22px; font-weight:900; }
    .metric-value-blue { color:#67A9FF; font-size:22px; font-weight:900; }
    .rec-check-item { display:flex; align-items:flex-start; gap:8px; color:#D6E4FF; font-size:12px; line-height:1.5; margin-bottom:7px; }
    .rec-check-icon { color:#34D399; font-size:14px; margin-top:1px; flex-shrink:0; }
    .threshold-impact-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-top:10px; }
    .impact-box { border-radius:10px; padding:10px 8px; text-align:center; }
    .impact-box-low  { background:rgba(220,38,38,0.12);  border:1px solid rgba(220,38,38,0.28); }
    .impact-box-mid  { background:rgba(52,211,153,0.10); border:1px solid rgba(52,211,153,0.30); }
    .impact-box-high { background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.28); }
    .impact-title { font-size:11px; font-weight:800; margin-bottom:5px; }
    .impact-desc  { color:#B8C9E8; font-size:10px; line-height:1.4; }
    .sim-tip { padding:9px 11px; border-radius:10px; margin-top:4px;
        background:rgba(96,165,250,0.05); border:1px solid rgba(96,165,250,0.18);
        color:#D6E4FF; font-size:11px; line-height:1.5; }
    .panel-header { display:flex; align-items:center; gap:8px; margin-bottom:2px; }
    .panel-badge { width:24px; height:24px; border-radius:999px; flex-shrink:0; display:flex;
        align-items:center; justify-content:center; background:rgba(66,111,189,0.35);
        border:1px solid rgba(125,163,216,0.28); color:#EAF2FF; font-size:12px; font-weight:900; }
    .panel-title { color:#EAF2FF; font-size:13px; font-weight:900; }
    .panel-info  { color:#7DA4D8; font-size:13px; margin-left:auto; }
    .panel-sub   { color:#B8C9E8; font-size:11px; font-weight:600; margin:2px 0 6px 32px; }
    .sim-section-gap { height:16px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        '<div style="font-size:30px;font-weight:900;color:#FFFFFF;">Simulation</div>'
        '<div style="display:inline-flex;align-items:center;justify-content:center;min-width:46px;height:24px;'
        'padding:0 9px;border-radius:999px;background:rgba(114,155,255,0.12);'
        'border:1px solid rgba(114,155,255,0.28);color:#9EBBFF;font-size:12px;font-weight:800;">v2.0</div>'
        '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown(
                '<div style="color:#C6D8F5;font-size:12px;font-weight:700;margin-bottom:10px;">'
                '모델 : XGBoost — 테스트 데이터 실제 예측 기반</div>'
                '<div class="sim-summary-grid">'
                f'<div style="padding:10px 8px 10px 0;border-right:1px solid rgba(148,163,184,0.14);">'
                f'<div style="color:#7DA4D8;font-size:11px;font-weight:700;margin-bottom:4px;">전체 거래</div>'
                f'<div style="color:#DCE9FF;font-size:17px;font-weight:900;">{_fmt_int(total_tx)}건</div></div>'
                f'<div style="padding:10px 8px;border-right:1px solid rgba(148,163,184,0.14);">'
                f'<div style="color:#7DA4D8;font-size:11px;font-weight:700;margin-bottom:4px;">실제 Fraud</div>'
                f'<div style="color:#FF8A8A;font-size:17px;font-weight:900;">{_fmt_int(fraud_tx)}건</div></div>'
                f'<div style="padding:10px 8px;border-right:1px solid rgba(148,163,184,0.14);">'
                f'<div style="color:#7DA4D8;font-size:11px;font-weight:700;margin-bottom:4px;">Fraud 비율</div>'
                f'<div style="color:#FFB455;font-size:17px;font-weight:900;">{fraud_tx/total_tx*100:.2f}%</div></div>'
                f'<div style="padding:10px 8px;border-right:1px solid rgba(148,163,184,0.14);">'
                f'<div style="color:#7DA4D8;font-size:11px;font-weight:700;margin-bottom:4px;">ROC-AUC</div>'
                f'<div style="color:#64D39D;font-size:17px;font-weight:900;">{roc_auc:.4f}</div></div>'
                f'<div style="padding:10px 0 10px 8px;">'
                f'<div style="color:#7DA4D8;font-size:11px;font-weight:700;margin-bottom:4px;">PR-AUC</div>'
                f'<div style="color:#9C8CFF;font-size:17px;font-weight:900;">{pr_auc:.4f}</div></div>'
                '</div>', unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            h1, h2, h3 = st.columns([2.2, 1, 1.2])
            with h1:
                st.markdown('<div style="color:#DCE9FF;font-size:13px;font-weight:800;padding-top:2px;">Threshold 조정 (0.01 ~ 0.99)</div>', unsafe_allow_html=True)
            with h2:
                cur_val = st.session_state.get("threshold_slider_widget", default_threshold)
                st.markdown(f'<div style="text-align:center;color:#7EE081;font-size:22px;font-weight:900;line-height:1;">{cur_val:.2f}</div>', unsafe_allow_html=True)
            with h3:
                if st.button("↺ 기본값으로 초기화", key="reset_threshold"):
                    st.session_state["threshold_val"] = default_threshold
                    st.session_state["threshold_slider_widget"] = default_threshold
                    st.rerun()

            threshold = st.slider("threshold_slider", min_value=0.01, max_value=0.99,
                                  step=0.01, format="%.2f",
                                  label_visibility="collapsed", key="threshold_slider_widget")
            st.session_state["threshold_val"] = threshold
            threshold = round(float(threshold), 2)
            st.markdown(f'<div style="display:flex;justify-content:space-between;color:#7EA3CF;font-size:11px;font-weight:700;margin-top:-6px;"><span>0.01</span><span>0.99</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    tn = total_tx - fraud_tx - fp
    best_metrics = compute_metrics_at_threshold(y_true, y_prob, best_t)
    best_fp = best_metrics["fp"]

    kpi_html = '<div class="sim-kpi-grid">' + "".join([
        _render_kpi_card("Precision", _fmt_float(metrics["precision"]), "#67A9FF"),
        _render_kpi_card("Recall",    _fmt_float(metrics["recall"]),    "#FFB14E"),
        _render_kpi_card("F1-Score",  _fmt_float(metrics["f1"]),        "#FF87D1"),
        _render_kpi_card("F2-Score",  _fmt_float(metrics["f2"]),        "#B292FF"),
        _render_kpi_card("탐지 (TP)", _fmt_int(tp),                     "#72E4A8"),
        _render_kpi_card("오탐 (FP)", _fmt_int(fp),                     "#FF8A8A"),
    ]) + '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown('<div class="sim-section-gap"></div>', unsafe_allow_html=True)

    x = df_table["threshold"].values
    chart_meta = [
        ("1","Threshold별 지표 변화",         "Threshold에 따른 주요 지표 변화",          "Threshold를 낮추면 Recall은 증가하지만 Precision은 감소합니다."),
        ("2","Trade-off (Precision vs Recall)","Precision과 Recall의 상충 관계",           "더 높은 Recall을 얻으려면 Precision이 희생될 수 있습니다."),
        ("3","탐지 (TP) vs 오탐 (FP) 변화",   "Threshold에 따른 탐지건수와 오탐건수 변화","Threshold를 낮추면 탐지는 증가하지만, 오탐도 증가합니다."),
        ("4","놓친 사기 (FN) 변화",            "Threshold에 따른 놓친 사기(FN) 건수 변화", "Threshold를 낮출수록 놓치는 사기 건수는 줄어듭니다."),
    ]
    chart_cols = st.columns(4, gap="medium")
    for idx, col in enumerate(chart_cols):
        step, title, subtitle, tip = chart_meta[idx]
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="panel-header"><div class="panel-badge">{step}</div><div class="panel-title">{title}</div><div class="panel-info">ⓘ</div></div><div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)
                if idx == 0:
                    fig = go.Figure()
                    for arr, name, color in [(df_table["precision"].values,"Precision","#5BA4FF"),
                                             (df_table["recall"].values,   "Recall",   "#FFB14E"),
                                             (df_table["f1"].values,       "F1-Score", "#FF7CC8"),
                                             (df_table["f2"].values,       "F2-Score", "#A88CFF")]:
                        fig.add_trace(go.Scatter(x=x, y=arr, name=name, mode="lines+markers",
                                                 line=dict(color=color, width=2.2), marker=dict(size=4)))
                    fig.add_vline(x=best_t, line_dash="solid", line_color="#34D399", line_width=2)
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#FDE68A", line_width=2)
                    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", **_chart_layout(245))
                elif idx == 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_table["recall"].values, y=df_table["precision"].values,
                                            name="PR Curve", mode="lines+markers",
                                            line=dict(color="#6FA8FF", width=2.4), marker=dict(size=4)))
                    fig.add_trace(go.Scatter(x=[metrics["recall"]], y=[metrics["precision"]], name="현재",
                                            mode="markers", marker=dict(size=12, color="#7EE081",
                                            line=dict(color="#E8FFE8", width=2))))
                    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", **_chart_layout(245))
                elif idx == 2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=df_table["tp"].values, name="탐지 (TP)", mode="lines", line=dict(color="#5DE3A0", width=2.4)))
                    fig.add_trace(go.Scatter(x=x, y=df_table["fp"].values, name="오탐 (FP)", mode="lines", line=dict(color="#FF7D77", width=2.4)))
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#EAF2FF", line_width=2)
                    fig.update_layout(xaxis_title="Threshold", yaxis_title="건수", **_chart_layout(245))
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=df_table["fn"].values, name="미탐 (FN)", mode="lines+markers",
                                            line=dict(color="#B48DFF", width=2.4), marker=dict(size=4)))
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#7EE081", line_width=2)
                    fig.update_layout(xaxis_title="Threshold", yaxis_title="건수", **_chart_layout(245))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f'<div class="sim-tip">💡 {tip}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sim-section-gap"></div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown(
                f'<div style="color:#FFFFFF;font-size:14px;font-weight:900;margin-bottom:10px;">'
                f'Confusion Matrix <span style="font-size:12px;color:#7DA4D8;">(Threshold = {threshold:.2f})</span></div>'
                f'<table class="cm-table">'
                f'<tr><th style="width:22%;">실제 ∖ 예측</th><th>사기 (Pos)</th><th>정상 (Neg)</th><th>합계</th></tr>'
                f'<tr><td class="label">사기 (실제 Fraud)</td><td class="tp">{_fmt_int(tp)} (TP)</td><td class="fn">{_fmt_int(fn)} (FN)</td><td class="total">{_fmt_int(fraud_tx)}</td></tr>'
                f'<tr><td class="label">정상 (실제 Normal)</td><td class="fp">{_fmt_int(fp)} (FP)</td><td class="tn">{_fmt_int(tn)} (TN)</td><td class="total">{_fmt_int(total_tx-fraud_tx)}</td></tr>'
                f'<tr><td class="label">합계</td><td class="total">{_fmt_int(tp+fp)}</td><td class="total">{_fmt_int(fn+tn)}</td><td class="total">{_fmt_int(total_tx)}</td></tr>'
                f'</table>'
                f'<div class="metric-box">'
                f'<div class="metric-item" style="background:rgba(220,38,38,0.10);border:1px solid rgba(220,38,38,0.28);"><div class="metric-label">TPR (Recall)</div><div class="metric-value-red">{metrics["recall"]:.4f}</div></div>'
                f'<div class="metric-item" style="background:rgba(220,38,38,0.10);border:1px solid rgba(220,38,38,0.28);"><div class="metric-label">FPR</div><div class="metric-value-red">{fp/(total_tx-fraud_tx) if (total_tx-fraud_tx)>0 else 0:.4f}</div></div>'
                f'<div class="metric-item"><div class="metric-label">TNR (Specificity)</div><div class="metric-value-blue">{tn/(total_tx-fraud_tx) if (total_tx-fraud_tx)>0 else 0:.4f}</div></div>'
                f'<div class="metric-item"><div class="metric-label">FNR (Miss Rate)</div><div class="metric-value-blue">{fn/fraud_tx if fraud_tx>0 else 0:.4f}</div></div>'
                f'</div>', unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown(
                f'<div style="color:#FFFFFF;font-size:14px;font-weight:900;margin-bottom:12px;">'
                f'추천 설정 분석 <span style="font-size:12px;color:#7DA4D8;">(Threshold = {best_t:.2f})</span></div>'
                f'<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;">'
                f'<div style="background:rgba(52,211,153,0.12);border:2px solid rgba(52,211,153,0.35);border-radius:14px;padding:14px 16px;text-align:center;min-width:88px;">'
                f'<div style="color:#34D399;font-size:11px;font-weight:800;margin-bottom:4px;">추천 Threshold</div>'
                f'<div style="color:#34D399;font-size:32px;font-weight:900;line-height:1;">{best_t:.2f}</div>'
                f'<div style="color:#34D399;font-size:10px;font-weight:700;margin-top:4px;">(F2 기준 최적)</div></div>'
                f'<div style="flex:1;">'
                f'<div class="rec-check-item"><span class="rec-check-icon">✅</span><span>사기 탐지와 오탐 간 균형이 가장 좋은 구간입니다.</span></div>'
                f'<div class="rec-check-item"><span class="rec-check-icon">✅</span><span>Recall {best_metrics["recall"]*100:.1f}% 확보로 사기 탐지 성능이 우수합니다.</span></div>'
                f'<div class="rec-check-item"><span class="rec-check-icon">✅</span><span>오탐 {best_fp:,}건 수준으로 고객 피해를 최소화하고 있습니다.</span></div>'
                f'<div class="rec-check-item"><span class="rec-check-icon">✅</span><span>F2-Score {best_metrics["f2"]:.4f}로 Recall에 높은 가중치를 둔 최적의 선택입니다.</span></div>'
                f'</div></div>'
                f'<div style="color:#A8C0E0;font-size:11px;font-weight:800;margin-bottom:6px;">Threshold 조정 시 영향</div>'
                f'<div class="threshold-impact-grid">'
                f'<div class="impact-box impact-box-low"><div class="impact-title" style="color:#FF8A8A;">더 낮추면 (&lt; {best_t:.2f})</div><div class="impact-desc">탐지↑ (Recall↑) | 오탐↑ (FP↑)<br>놓친 사기↓ (FN↓)</div></div>'
                f'<div class="impact-box impact-box-mid"><div class="impact-title" style="color:#34D399;">유지하면 (= {best_t:.2f})</div><div class="impact-desc">탐지와 오탐 균형 유지<br>F2 기준 최적 성능 유지</div></div>'
                f'<div class="impact-box impact-box-high"><div class="impact-title" style="color:#67A9FF;">더 높이면 (&gt; {best_t:.2f})</div><div class="impact-desc">탐지↓ (Recall↓) | 오탐↓ (FP↓)<br>놓친 사기↑ (FN↑)</div></div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;">'
        '<div style="color:#7F9AC2;font-size:11px;font-weight:700;">※ 비용 가정: 미탐 비용(FN) 10,000원 / 오탐 비용(FP) 1,000원 (예시)</div>'
        '<div style="color:#7F9AC2;font-size:11px;font-weight:700;display:flex;gap:14px;">'
        '<span>모델 : XGBoost</span><span>|</span><span>데이터 : 테스트 세트</span></div></div>',
        unsafe_allow_html=True)
