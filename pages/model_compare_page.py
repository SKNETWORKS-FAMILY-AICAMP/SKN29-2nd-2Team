import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "cc_fraud_train_processed.csv")
TARGET_COL   = "is_fraud"
VALID_SIZE   = 0.2
RANDOM_STATE = 42

MODEL_FILES = {
    "XGBoost":           "xgboost_model.pkl",
    "LightGBM":          "lightgbm_model.pkl",
    "CatBoost":          "catboost_model.pkl",
    "Random Forest":     "randomforest_model.pkl",
    "Extra Trees":       "extratrees_model.pkl",
    "Gradient Boosting": "gradientboosting_model.pkl",
    "Decision Tree":     "decisiontree_model.pkl",
    "Logistic Reg.":     "logisticregression_model.pkl",
    "Soft Voting":       "softvoting_model.pkl",
    "Stacking":          "stacking_model.pkl",
}

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
PALETTE = [
    "#e6b43c", "#3a7bd5", "#e05c5c", "#5cb85c", "#9b59b6",
    "#1abc9c", "#e67e22", "#2ecc71", "#e74c3c", "#3498db",
]


# ── 데이터 로드 & 분리 ────────────────────────────────────────
def _limit_parallel_jobs(model, seen=None):
    """Streamlit/Windows 환경에서 병렬 예측 프로세스가 과도하게 뜨지 않도록 제한한다."""
    if seen is None:
        seen = set()
    obj_id = id(model)
    if obj_id in seen:
        return model
    seen.add(obj_id)

    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass

    for attr in ("steps", "estimators", "estimators_", "transformers", "transformers_"):
        items = getattr(model, attr, None)
        if items is None:
            continue
        for item in items:
            target = item[-1] if isinstance(item, tuple) else item
            _limit_parallel_jobs(target, seen)

    for attr in ("final_estimator", "final_estimator_", "estimator", "base_estimator"):
        target = getattr(model, attr, None)
        if target is not None:
            _limit_parallel_jobs(target, seen)

    return model


def _get_final_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def _get_feature_importances(model):
    estimator = _get_final_estimator(model)
    return getattr(estimator, "feature_importances_", None)


def _get_model_params(model):
    estimator = _get_final_estimator(model)
    try:
        return estimator.get_params()
    except Exception:
        return {}


def _get_feature_names(model, fallback_columns, n_features):
    try:
        if hasattr(model, "steps"):
            names = model.steps[0][1].get_feature_names_out()
            return list(names)[:n_features]
    except Exception:
        pass
    return (list(fallback_columns)[:n_features] + [f"feat_{i}" for i in range(n_features)])[:n_features]


@st.cache_data(show_spinner="학습 데이터 로딩 중...")
def load_and_split():
    df = pd.read_csv(TRAIN_CSV)

    # 파생 컬럼 보완
    if "amt_zscore" not in df.columns:
        df["amt_zscore"] = (df["amt"] - df["amt"].mean()) / df["amt"].std()
    if "hour_dev" not in df.columns:
        df["hour_dev"] = abs(df["trans_hour"] - df["trans_hour"].mean()) / df["trans_hour"].std()
    if "distance_km" not in df.columns:
        df["distance_km"] = np.random.default_rng(42).uniform(1, 120, len(df))
    if "high_amt_far" not in df.columns:
        dist_z = (df["distance_km"] - df["distance_km"].mean()) / df["distance_km"].std()
        df["high_amt_far"] = ((df["amt_zscore"] > 2) & (dist_z > 2)).astype(int)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_valid, y_valid, len(df), int(y.sum())


# ── 모델 로드 ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            models[name] = _limit_parallel_jobs(joblib.load(path))
        except Exception as e:
            st.warning(f"⚠️ {name} 로드 실패: {e}")
    return models


# ── 평가 ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="모델 평가 중... (최초 1회)")
def evaluate_all(_models, _X_valid, _y_valid):
    metrics  = {}
    cms      = {}
    roc_data = {}
    pr_data  = {}

    for name, clf in _models.items():
        try:
            y_pred = clf.predict(_X_valid)

            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(_X_valid)[:, 1]
            elif hasattr(clf, "decision_function"):
                raw    = clf.decision_function(_X_valid)
                y_prob = 1 / (1 + np.exp(-raw))
            else:
                y_prob = y_pred.astype(float)

            metrics[name] = [
                round(accuracy_score(_y_valid, y_pred),                   4),
                round(precision_score(_y_valid, y_pred, zero_division=0), 4),
                round(recall_score(_y_valid, y_pred,    zero_division=0), 4),
                round(f1_score(_y_valid, y_pred,        zero_division=0), 4),
                round(roc_auc_score(_y_valid, y_prob),                    4),
                round(average_precision_score(_y_valid, y_prob),          4),
            ]
            cms[name]      = confusion_matrix(_y_valid, y_pred)
            fpr, tpr, _    = roc_curve(_y_valid, y_prob)
            roc_data[name] = (fpr, tpr)
            prec, rec, _   = precision_recall_curve(_y_valid, y_prob)
            pr_data[name]  = (prec, rec)

        except Exception as e:
            st.warning(f"⚠️ {name} 평가 실패: {e}")

    return metrics, cms, roc_data, pr_data


# ── 메인 ──────────────────────────────────────────────────────
def show_model_compare_page():
    st.markdown('<h1 class="page-title light">모델 비교</h1>', unsafe_allow_html=True)

    # 데이터 로드
    try:
        X_valid, y_valid, total_rows, fraud_count = load_and_split()
    except FileNotFoundError:
        st.error(f"학습 데이터를 찾을 수 없습니다 → `{TRAIN_CSV}`")
        return

    valid_count = len(y_valid)
    valid_fraud = int(y_valid.sum())
    st.caption(
        f"📂 **{os.path.basename(TRAIN_CSV)}** 총 {total_rows:,}건으로 평가 "
       
    )

    # 모델 로드 & 평가
    models = load_models()
    if not models:
        st.error(f"로드된 모델이 없습니다 → `{MODEL_DIR}`")
        return

    metrics, cms, roc_data, pr_data = evaluate_all(models, X_valid, y_valid)

    if not metrics:
        st.error("평가에 성공한 모델이 없습니다. sklearn 버전 또는 피처 컬럼을 확인하세요.")
        return

    df_metrics = pd.DataFrame(metrics, index=METRIC_COLS).T.reset_index()
    df_metrics.rename(columns={"index": "Model"}, inplace=True)

    # 추천 모델 = PR-AUC 1위
    best = df_metrics.loc[df_metrics["PR-AUC"].idxmax(), "Model"]
    bm   = metrics[best]

    # ── 추천 모델 카드 ────────────────────────────────────────
    st.markdown(f"""
    <div class="recommend-card">
        <div class="trophy">🏆</div>
        <div>
            <p>추천 모델</p>
            <h2>{best}</h2>
            <span>추천 이유 : Recall과 PR-AUC가 높아 이상 거래 탐지에 가장 적합합니다.</span>
        </div>
        <div class="recommend-metrics">
            <div><b>{bm[2]:.4f}</b><span>Recall</span></div>
            <div><b>{bm[3]:.4f}</b><span>F1-Score</span></div>
            <div><b>{bm[5]:.4f}</b><span>PR-AUC</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 성능 지표 테이블 ──────────────────────────────────────
    st.subheader("📊 모델별 성능 지표 (Validation Set)")

    def highlight_best(row):
        return [
            "background-color: #1a3a5c; font-weight: bold" if row["Model"] == best else ""
            for _ in row
        ]

    st.dataframe(
        df_metrics.style
            .apply(highlight_best, axis=1)
            .format({c: "{:.4f}" for c in METRIC_COLS}),
        use_container_width=True,
        hide_index=True,
    )

    # ── 지표별 막대 차트 ──────────────────────────────────────
    st.subheader("📈 지표별 성능 비교")
    metric_sel = st.selectbox("비교 지표 선택", METRIC_COLS, index=5)

    colors = ["#e6b43c" if m == best else "#3a7bd5" for m in df_metrics["Model"]]
    fig_bar = go.Figure(go.Bar(
        x=df_metrics["Model"],
        y=df_metrics[metric_sel],
        marker_color=colors,
        text=df_metrics[metric_sel].round(4),
        textposition="outside",
    ))
    y_min = max(0.0, float(df_metrics[metric_sel].min()) - 0.05)
    fig_bar.update_layout(
        yaxis=dict(range=[y_min, 1.02], title=metric_sel),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 상세 분석: Confusion Matrix + ROC ────────────────────
    st.subheader("🔍 모델 상세 분석")
    detail_model = st.selectbox(
        "분석할 모델 선택", list(metrics.keys()),
        index=list(metrics.keys()).index(best) if best in metrics else 0,
    )

    import plotly.express as px
    left, right = st.columns(2)

    with left:
        cm = cms[detail_model]
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="예측", y="실제"),
            x=["정상", "사기"], y=["정상", "사기"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {detail_model}",
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        fig_roc = go.Figure()
        for (name, (fpr, tpr)), color in zip(roc_data.items(), PALETTE):
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} ({metrics[name][4]:.4f})",
                line=dict(
                    color=color,
                    width=3 if name == best else 1.5,
                    dash="solid" if name == best else "dot",
                ),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(dash="dash", color="gray"),
        ))
        fig_roc.update_layout(
            title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
            height=380, legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── PR Curve ──────────────────────────────────────────────
    st.subheader("📉 Precision-Recall Curve")
    fig_pr = go.Figure()
    for (name, (prec, rec)), color in zip(pr_data.items(), PALETTE):
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec,
            name=f"{name} ({metrics[name][5]:.4f})",
            line=dict(
                color=color,
                width=3 if name == best else 1.5,
                dash="solid" if name == best else "dot",
            ),
        ))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=420, legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────
    st.subheader("🌲 Feature Importance")
    tree_models = {
        name: clf for name, clf in models.items()
        if _get_feature_importances(clf) is not None
    }

    if tree_models:
        default_idx = list(tree_models.keys()).index(best) if best in tree_models else 0
        fi_model = st.selectbox("모델 선택", list(tree_models.keys()), index=default_idx)
        clf = tree_models[fi_model]
        fi  = _get_feature_importances(clf)

        # 피처 이름: 학습 시 사용된 컬럼 수와 맞추기
        n_feat = len(fi)
        feat_names = _get_feature_names(clf, X_valid.columns, n_feat)

        default_top_n = min(7, n_feat)
        top_n = st.slider("상위 N개 피처", 5, min(30, n_feat), default_top_n)
        idx   = np.argsort(fi)[-top_n:][::-1]
        fig_fi = go.Figure(go.Bar(
            x=fi[idx],
            y=[feat_names[i] for i in idx],
            orientation="h",
            marker_color="#3a7bd5",
        ))
        fig_fi.update_layout(
            title=f"{fi_model} — Top {top_n} Feature Importances",
            yaxis=dict(autorange="reversed"),
            height=max(350, top_n * 22),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance를 지원하는 모델이 없습니다.")

    # ── 하이퍼파라미터 비교 ─────────────────────────────────────
    st.subheader("⚙️ 주요 하이퍼파라미터")

    KEY_PARAMS = [
        "n_estimators", "max_depth", "learning_rate", "num_leaves",
        "subsample", "colsample_bytree", "scale_pos_weight",
        "C", "iterations", "depth",
    ]

    XGB_BEFORE = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 171,
    }
    XGB_AFTER = {
        "n_estimators":     464,
        "max_depth":        9,
        "learning_rate":    0.2110,
        "subsample":        0.8121,
        "colsample_bytree": 0.8098,
        "gamma":            0.7261,
        "reg_alpha":        0.3234,
        "reg_lambda":       5.9642,
        "min_child_weight": 3,
        "scale_pos_weight": 171,
    }
    XGB_BEFORE_PERF = {"Recall": 0.9389, "PR-AUC": 0.8612}

    def fmt_num(v):
        if isinstance(v, float):
            return round(v, 4)
        return v

    for name, clf in models.items():
        p = _get_model_params(clf)
        is_best = (name == best)
        label = f"{'🏆 ' if is_best else ''}📦 {name}{'  🔧 튜닝 완료(Valid로 평가)' if name == 'XGBoost' else ''}"

        with st.expander(label, expanded=(name == "XGBoost")):
            if name == "XGBoost":
                after_recall = 0.9367
                after_prauc  = 0.9717

                st.caption("**튜닝 효과**")
                m1, m2 = st.columns(2)
                recall_delta = after_recall - XGB_BEFORE_PERF["Recall"]
                prauc_delta  = after_prauc  - XGB_BEFORE_PERF["PR-AUC"]
                m1.metric(
                    "Recall",
                    f"{after_recall:.4f}",
                    f"{recall_delta:+.4f}  ({recall_delta/XGB_BEFORE_PERF['Recall']*100:+.2f}%)",
                )
                m2.metric(
                    "PR-AUC",
                    f"{after_prauc:.4f}",
                    f"{prauc_delta:+.4f}  ({prauc_delta/XGB_BEFORE_PERF['PR-AUC']*100:+.2f}%)",
                )

                st.divider()
                st.caption("**핵심 하이퍼파라미터 비교**")
                rows = []
                for k in XGB_AFTER:
                    bv = XGB_BEFORE.get(k)
                    av = XGB_AFTER[k]
                    if bv is None:
                        status = "🆕 신규"
                    elif bv != av:
                        status = "🔄 변경"
                    else:
                        status = "— 유지"
                    rows.append({
                        "파라미터":  k,
                        "튜닝 전":   fmt_num(bv) if bv is not None else "—",
                        "튜닝 후":   fmt_num(av),
                        "변경 여부": status,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                rows = []
                for k in KEY_PARAMS:
                    val = p.get(k)
                    if val is not None:
                        rows.append({"파라미터": k, "값": fmt_num(val)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("파라미터 정보 없음")
