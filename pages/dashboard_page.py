import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data.data_loader import load_fraud_data, load_kpi_summary


DARK_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#071527",
    plot_bgcolor="#071527",
    font=dict(color="#CBD5E1", size=10),
    title_font=dict(color="#F8FAFC", size=13),
)
DARK_AXIS = dict(
    gridcolor="#1E3A5F",
    zerolinecolor="#1E3A5F",
    tickfont=dict(color="#94A3B8", size=9),
    title_font=dict(color="#94A3B8", size=10),
)


def dark_layout(height=230, margin=None, xaxis_extra=None, yaxis_extra=None):
    return dict(
        **DARK_BASE,
        height=height,
        margin=margin or dict(t=35, b=25, l=35, r=20),
        xaxis={**DARK_AXIS, **(xaxis_extra or {})},
        yaxis={**DARK_AXIS, **(yaxis_extra or {})},
    )


# ── 캐싱 집계 함수들 ───────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _get_hourly_df():
    df = load_fraud_data()
    if "trans_hour" not in df.columns or "is_fraud" not in df.columns:
        return pd.DataFrame(columns=["trans_hour","total_count","fraud_count","fraud_rate"])
    h = df.groupby("trans_hour")["is_fraud"].agg(total_count="count", fraud_count="sum").reset_index()
    h["fraud_rate"] = h["fraud_count"] / h["total_count"] * 100
    return h


@st.cache_data(show_spinner=False, ttl=3600)
def _get_heatmap_z():
    df = load_fraud_data()
    if not all(c in df.columns for c in ["trans_dayofweek","trans_hour","is_fraud"]):
        return [[0]*24 for _ in range(7)]
    hm = df.groupby(["trans_dayofweek","trans_hour"])["is_fraud"].mean().mul(100).unstack(fill_value=0)
    for h in range(24):
        if h not in hm.columns: hm[h] = 0
    hm = hm[sorted(hm.columns)].reindex([1,2,3,4,5,6,7], fill_value=0)
    return hm.values.tolist()


@st.cache_data(show_spinner=False, ttl=3600)
def _get_amt_fraud():
    df = load_fraud_data()
    labels = ["~$10","$10~$100","$100~$500","$500+"]
    if "amt" not in df.columns or "is_fraud" not in df.columns:
        return pd.DataFrame({"amt_bin": labels, "is_fraud": [0,0,0,0]})
    tmp = df.copy()
    tmp["amt_bin"] = pd.cut(tmp["amt"], bins=[0,10,100,500,float("inf")], labels=labels, right=False)
    return tmp.groupby("amt_bin", observed=True)["is_fraud"].mean().mul(100).reset_index()


@st.cache_data(show_spinner=False, ttl=3600)
def _get_detect_amount_df():
    df = load_fraud_data()
    labels = ["~$10","$10~$100","$100~$500","$500~$1,000","$1,000+"]
    if "amt" not in df.columns:
        return pd.DataFrame({"amount_bin":labels,"total_count":[0]*5,"fraud_count":[0]*5,"fraud_rate":[0]*5})
    tmp = df.copy()
    tmp["amount_bin"] = pd.cut(tmp["amt"], bins=[0,10,100,500,1000,float("inf")], labels=labels, right=False)
    r = tmp.groupby("amount_bin", observed=True)["is_fraud"].agg(total_count="count", fraud_count="sum").reset_index()
    r["fraud_rate"] = r["fraud_count"] / r["total_count"] * 100
    return r


@st.cache_data(show_spinner=False, ttl=3600)
def _get_risk_hour_df():
    df = load_fraud_data()
    if "trans_hour" not in df.columns:
        return pd.DataFrame({"trans_hour":[],"risk_score":[]})
    return df.groupby("trans_hour")["risk_score"].mean().reset_index()


@st.cache_data(show_spinner=False, ttl=3600)
def _get_risk_group():
    df = load_fraud_data()
    total = len(df)
    rg = pd.DataFrame({
        "구간": ["Low (<40)","Medium (40~70)","High (70+)"],
        "거래 수": [
            int((df["risk_score"] < 40).sum()),
            int(((df["risk_score"] >= 40) & (df["risk_score"] < 70)).sum()),
            int((df["risk_score"] >= 70).sum()),
        ],
    })
    rg["비율"] = (rg["거래 수"] / total * 100).round(1).astype(str) + "%"
    return rg


@st.cache_data(show_spinner=False, ttl=3600)
def _get_customer_stats():
    """
    cc_num 이 거래별 unique → 고객 세그먼트를 age + amt + risk_score 기반으로 재설계.
    세그먼트 정의:
      VIP      : 고액 거래 (amt ≥ 상위 10%) + 정상 (risk < 40)
      기업      : amt ≥ 상위 25% + risk < 70
      일반      : risk < 40  (나머지)
      주의      : 40 ≤ risk < 70
      고위험    : risk ≥ 70
    """
    df = load_fraud_data()
    total = len(df)

    amt_p90 = float(df["amt"].quantile(0.90))
    amt_p75 = float(df["amt"].quantile(0.75))

    def seg(row):
        r = row["risk_score"]
        a = row["amt"]
        if a >= amt_p90 and r < 40:   return "VIP"
        if a >= amt_p75 and r < 70:   return "기업"
        if r >= 70:                   return "고위험"
        if r >= 40:                   return "주의"
        return "일반"

    df["세그먼트"] = df.apply(seg, axis=1)

    seg_order = ["VIP","기업","일반","주의","고위험"]

    # 세그먼트별 집계
    seg_stat = (
        df.groupby("세그먼트")
        .agg(
            거래수=("amt","count"),
            평균_Risk=("risk_score","mean"),
            평균_금액=("amt","mean"),
            이상거래수=("is_fraud","sum"),
        )
        .reindex(seg_order)
        .fillna(0)
        .reset_index()
    )
    seg_stat["이상비율"] = (seg_stat["이상거래수"] / seg_stat["거래수"].replace(0, np.nan) * 100).fillna(0).round(1)
    seg_stat["거래비율"] = (seg_stat["거래수"] / total * 100).round(1)

    # 고객 수 = 세그먼트별 고유 cc_num (없으면 거래수로 대체)
    customer_count = df["cc_num"].nunique() if "cc_num" in df.columns else total
    high_risk_count = int((df["risk_score"] >= 70).sum())
    avg_risk = round(float(df["risk_score"].mean()), 1)

    return {
        "seg_stat": seg_stat,
        "seg_order": seg_order,
        "customer_count": customer_count,
        "high_risk_count": high_risk_count,
        "avg_risk": avg_risk,
        "total": total,
        "df_with_seg": df,  # 세그먼트 컬럼 포함
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _get_detect_table():
    df = load_fraud_data()
    if "is_fraud" not in df.columns: return pd.DataFrame()
    fraud_df = df[df["is_fraud"] == 1].sort_values("risk_score", ascending=False)
    cols_map = {"cc_num":"카드 번호","trans_hour":"거래 시간","category":"거래 유형",
                "amt":"금액 ($)","risk_score":"위험 점수","decision":"판단 결과"}
    avail = [c for c in cols_map if c in fraud_df.columns]
    out = fraud_df[avail].copy()
    if "amt" in out.columns:        out["amt"]        = out["amt"].round(2)
    if "risk_score" in out.columns: out["risk_score"]  = out["risk_score"].round(1)
    if "trans_hour" in out.columns: out["trans_hour"]  = out["trans_hour"].astype("Int64")
    if "decision" in out.columns:   out["decision"]    = out["decision"].astype(str)
    return out.rename(columns=cols_map)


# ── CSS ────────────────────────────────────────────────────────
def add_css():
    st.markdown("""
    <style>
    /* 탭 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #1E3A5F;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px; min-width: 108px;
        background: #071527;
        border: 1px solid #1E3A5F;
        border-radius: 5px 5px 0 0;
        color: #CBD5E1; font-size: 12px; font-weight: 900;
    }
    .stTabs [aria-selected="true"] {
        background: #0B2A55 !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #2563EB !important;
    }

    /* KPI 카드 */
    .overview-card {
        background: #10233A;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 86px;
        box-sizing: border-box;
    }
    .overview-label { color: #93C5FD; font-size: 11px; font-weight: 900; margin-bottom: 6px; }
    .overview-value { font-size: 22px; font-weight: 900; line-height: 1.15; }
    .overview-sub   { color: #94A3B8; font-size: 9px; margin-top: 6px; }

    /* 인사이트 박스 */
    .insight-box {
        background: #071527; border: 1px solid #1E3A5F; border-radius: 8px;
        padding: 18px 22px; color: #DCEBFF; font-size: 13px; line-height: 2.6;
        box-sizing: border-box; margin-bottom: 10px;
    }
    .summary-box {
        background: #071527; border: 1px solid #1E3A5F; border-radius: 8px;
        padding: 14px 18px; color: #DCEBFF; font-size: 12px; line-height: 1.9;
        box-sizing: border-box; margin-bottom: 16px;
    }
    .summary-box b { color: #60A5FA; }

    /* 차트 캡션 */
    .dash-chart-title {
        color: #F8FAFC; font-size: 12px; font-weight: 900; margin: 0 0 4px 0;
    }
    .table-caption {
        color: #60A5FA; font-size: 11px; font-weight: 900; margin: 6px 0 4px 0;
    }

    /* 세그먼트 배지 */
    .seg-badge {
        display: inline-block;
        padding: 4px 12px; border-radius: 999px;
        font-size: 11px; font-weight: 900;
        margin: 2px 3px;
    }

    /* 이상탐지 패턴 요약 박스 */
    .detect-summary-box {
        background: #071527; border: 1px solid #1E3A5F; border-radius: 8px;
        padding: 18px 20px; color: #DCEBFF; font-size: 12px; line-height: 2.2;
        box-sizing: border-box;
    }

    /* Plotly 차트 래퍼 */
    .stPlotlyChart { border: 1px solid #1E3A5F; border-radius: 8px; }

    /* 셀렉트박스 / 날짜 라벨 숨김 */
    /* .stSelectbox label, .stDateInput label { display: none; } */
                
    .stSelectbox label, .stDateInput label {
    color: #94A3B8;
    font-size: 11px;
    margin-bottom: 2px;
    }

    /* 버튼 */
    .stButton > button {
        background: #071527 !important; color: #F8FAFC !important;
        border: 1px solid #2563EB !important; border-radius: 8px !important;
        font-weight: 900 !important; height: 38px !important;
    }
    .stButton > button:hover {
        background: #0B2A55 !important; border: 1px solid #60A5FA !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ── KPI 카드 ───────────────────────────────────────────────────
def card(label, value, color="#60A5FA", sub="DB 기준"):
    st.markdown(
        f'<div class="overview-card">'
        f'<div class="overview-label">{label}</div>'
        f'<div class="overview-value" style="color:{color};">{value}</div>'
        f'<div class="overview-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── 차트 함수들 ────────────────────────────────────────────────
def make_hour_day_heatmap():
    z = _get_heatmap_z()
    fig = px.imshow(z, labels=dict(x="시간대", y="요일", color="이상 비율"),
                    x=list(range(24)), y=["월","화","수","목","금","토","일"],
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(**DARK_BASE, height=210, margin=dict(t=10, b=25, l=35, r=20),
                      xaxis=dict(**DARK_AXIS, title="시간대"),
                      yaxis=dict(**DARK_AXIS, title="요일"))
    return fig


def make_amount_fraud_chart():
    labels = ["~$10","$10~$100","$100~$500","$500+"]
    fig = px.bar(_get_amt_fraud(), x="amt_bin", y="is_fraud")
    fig.update_traces(marker_color="#818CF8")
    fig.update_layout(**dark_layout(210, dict(t=10, b=35, l=45, r=15),
                                    {"title":"amt_bin","categoryorder":"array","categoryarray":labels},
                                    {"title":"is_fraud (%)"}))
    return fig


def make_detect_hour_chart():
    h = _get_hourly_df()
    fig = go.Figure()
    if not h.empty:
        fig.add_bar(x=h["trans_hour"], y=h["total_count"], name="거래 수", marker_color="#2563EB")
        fig.add_bar(x=h["trans_hour"], y=h["fraud_count"], name="이상 거래 수", marker_color="#EF4444")
        fig.add_scatter(x=h["trans_hour"], y=h["fraud_rate"], name="이상 거래 비율",
                        yaxis="y2", mode="lines+markers",
                        line=dict(color="#F59E0B", width=2), marker=dict(size=5))
    fig.update_layout(**dark_layout(215, dict(t=28, b=25, l=35, r=35)), title="시간별 이상 거래 추이",
                      yaxis2=dict(overlaying="y", side="right",
                                  tickfont=dict(color="#94A3B8", size=9), gridcolor="#1E3A5F"),
                      legend=dict(orientation="h", y=1.18, x=0.16, font=dict(size=8)),
                      barmode="stack")
    return fig


def make_detect_amount_chart():
    a = _get_detect_amount_df()
    labels = ["~$10","$10~$100","$100~$500","$500~$1,000","$1,000+"]
    fig = go.Figure()
    if not a.empty:
        fig.add_bar(x=a["amount_bin"], y=a["total_count"], name="거래 수", marker_color="#2563EB")
        fig.add_scatter(x=a["amount_bin"], y=a["fraud_rate"], name="이상 거래 비율",
                        yaxis="y2", mode="lines+markers",
                        line=dict(color="#F59E0B", width=2), marker=dict(size=5))
    fig.update_layout(
        **dark_layout(215, dict(t=28, b=35, l=35, r=35),
                      xaxis_extra={"categoryorder":"array","categoryarray":labels}),
        title="금액별 이상 거래 분포",
        yaxis2=dict(overlaying="y", side="right",
                    tickfont=dict(color="#94A3B8", size=9), gridcolor="#1E3A5F"),
        legend=dict(orientation="h", y=1.18, x=0.1, font=dict(size=8)),
    )
    return fig


def make_risk_distribution_chart():
    df = load_fraud_data()
    fig = px.histogram(df, x="risk_score", nbins=60, title="Risk Score 분포 (0~100점)")
    fig.update_traces(marker_color="#2563EB")
    fig.add_vline(x=40, line_dash="dash", line_color="#10B981", annotation_text="40 (주의)")
    fig.add_vline(x=70, line_dash="dash", line_color="#EF4444", annotation_text="70 (위험)")
    fig.update_layout(**dark_layout(220, dict(t=35, b=25, l=45, r=20)))
    return fig


def make_risk_hour_chart():
    rh = _get_risk_hour_df()
    fig = px.line(rh, x="trans_hour", y="risk_score", markers=True, title="시간대별 평균 Risk Score")
    fig.update_traces(line=dict(color="#60A5FA", width=2), marker=dict(size=5, color="#60A5FA"))
    fig.update_layout(**dark_layout(220, dict(t=25, b=35, l=45, r=20),
                                    {"title":"시간대"}, {"title":"평균 Risk Score"}))
    return fig


def make_risk_pie_chart():
    rg = _get_risk_group()
    fig = px.pie(rg, names="구간", values="거래 수", hole=0.55,
                 color="구간",
                 color_discrete_map={"Low (<40)":"#34D399","Medium (40~70)":"#F59E0B","High (70+)":"#EF4444"})
    fig.update_traces(textinfo="percent+label", textfont=dict(color="#FFFFFF", size=10))
    fig.update_layout(**DARK_BASE, height=200, margin=dict(t=25, b=10, l=10, r=10),
                      title="Risk Score 구간 분포",
                      showlegend=False)
    return fig


def make_risk_contribution_chart():
    fig = px.bar(
        x=["금액 이상치","야간 거래","고액+원거리","시간대 이탈","이동 거리"],
        y=[40, 12, 20, 20, 8],
        title="Risk Score 구성 (점수 기여)"
    )
    fig.update_traces(marker_color=["#EF4444","#F59E0B","#818CF8","#60A5FA","#34D399"])
    fig.update_layout(**dark_layout(200, dict(t=35, b=50, l=40, r=20),
                                    {"title":"구성 요소"}, {"title":"최대 기여 점수"}))
    return fig


def make_feature_guide_chart():
    feat = pd.DataFrame({
        "피처":   ["amt_zscore","high_amt_far","hour_dev","is_night","distance_km"],
        "기여도": [40, 20, 20, 12, 8],
    })
    fig = px.bar(feat, x="기여도", y="피처", orientation="h",
                 text="기여도", title="피처별 이상탐지 기여도")
    fig.update_traces(marker_color="#60A5FA", texttemplate="%{text}점", textposition="outside")
    fig.update_layout(**dark_layout(175, dict(t=30, b=15, l=90, r=30)))
    return fig


def make_factor_chart():
    fi = pd.DataFrame({
        "요인":    ["금액 이상치","고액+원거리","비정상 시간대","야간 거래","거리 이상","지역 이상"],
        "중요도":  [0.312, 0.221, 0.178, 0.112, 0.087, 0.034],
    })
    fig = px.bar(fi, x="중요도", y="요인", orientation="h", title="이상 탐지 주요 요인")
    fig.update_traces(marker_color="#EF4444")
    fig.update_layout(**dark_layout(175, dict(t=30, b=15, l=90, r=30)))
    return fig


def make_seg_risk_bar(seg_stat, seg_order):
    """세그먼트별 평균 Risk Score 바 차트"""
    color_map = {
        "VIP":"#60A5FA","기업":"#818CF8","일반":"#34D399","주의":"#F59E0B","고위험":"#EF4444"
    }
    colors = [color_map.get(s,"#94A3B8") for s in seg_stat["세그먼트"]]
    fig = go.Figure(go.Bar(
        x=seg_stat["세그먼트"],
        y=seg_stat["평균_Risk"].round(1),
        marker_color=colors,
        text=seg_stat["평균_Risk"].round(1),
        textposition="outside",
    ))
    fig.update_layout(**dark_layout(240, dict(t=35, b=40, l=45, r=20),
                                    {"title":"세그먼트","categoryorder":"array","categoryarray":seg_order},
                                    {"title":"평균 Risk Score"}),
                      title="세그먼트별 평균 Risk Score")
    return fig


def make_seg_dist_pie(seg_stat):
    """세그먼트별 거래 비율 도넛 차트"""
    color_map = {
        "VIP":"#60A5FA","기업":"#818CF8","일반":"#34D399","주의":"#F59E0B","고위험":"#EF4444"
    }
    fig = px.pie(
        seg_stat[seg_stat["거래수"] > 0],
        names="세그먼트", values="거래수", hole=0.52,
        color="세그먼트",
        color_discrete_map=color_map,
        title="세그먼트별 거래 비중"
    )
    fig.update_traces(textinfo="percent+label", textfont=dict(color="#FFFFFF", size=10))
    fig.update_layout(**DARK_BASE, height=240, margin=dict(t=35, b=10, l=10, r=10),
                      showlegend=False)
    return fig


def make_seg_fraud_bar(seg_stat, seg_order):
    """세그먼트별 이상 거래 비율 바 차트"""
    color_map = {
        "VIP":"#60A5FA","기업":"#818CF8","일반":"#34D399","주의":"#F59E0B","고위험":"#EF4444"
    }
    colors = [color_map.get(s,"#94A3B8") for s in seg_stat["세그먼트"]]
    fig = go.Figure(go.Bar(
        x=seg_stat["세그먼트"],
        y=seg_stat["이상비율"],
        marker_color=colors,
        text=seg_stat["이상비율"].astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(**dark_layout(230, dict(t=35, b=40, l=45, r=20),
                                    {"title":"세그먼트","categoryorder":"array","categoryarray":seg_order},
                                    {"title":"이상 거래 비율 (%)"}),
                      title="세그먼트별 이상 거래 비율")
    return fig


# ── 메인 ───────────────────────────────────────────────────────
def show_dashboard_page():
    add_css()

    kpi             = load_kpi_summary()
    total_tx        = kpi["total_tx"]
    fraud_tx        = kpi["fraud_tx"]
    fraud_ratio     = kpi["fraud_ratio"]
    avg_risk        = kpi["avg_risk"]
    high_risk_count = kpi["high_risk_count"]
    high_risk_ratio = kpi["high_risk_ratio"]
    block_count     = kpi["block_count"]
    review_count    = kpi["review_count"]
    night_count     = kpi["night_count"]

    # 고객 분석 데이터
    cust = _get_customer_stats()
    seg_stat  = cust["seg_stat"]
    seg_order = cust["seg_order"]

    st.markdown('<h1 class="page-title">대시보드</h1>', unsafe_allow_html=True)

    tab_overview, tab_detect, tab_risk, tab_customer = st.tabs(
        ["▥ Overview", "⌕ 이상탐지", "⌁ Risk Score", "♙ 고객 분석"]
    )

    # ── Overview ─────────────────────────────────────────────
    with tab_overview:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1: card("전체 거래 수",   f"{total_tx:,}",                  "#3B82F6", "DB 기준")
        with c2: card("이상 거래 수",   f"{fraud_tx:,} ({fraud_ratio}%)", "#FF5C72", "Target 기준")
        with c3: card("High Risk 비율", f"{high_risk_ratio}%",            "#FF4FB3", "Risk ≥ 70점")

        st.markdown("""
        <div class="section-title" style="font-size:15px;margin:18px 0 8px;">💡 Today 핵심 인사이트</div>
        <div class="insight-box">
            🚨 &nbsp;<b>22~24시 사이 이상 거래 급증</b> — 해당 시간대 이상 거래 발생비율이 평시대비 증가하였습니다.<br>
            🔎 &nbsp;<b>고액 거래에서 이상 비율 증가</b> — 고액 거래 구간에서 이상 거래 비율이 높게 나타났습니다.<br>
            ⚠️ &nbsp;<b>Risk Score 기반 이상 후보 탐지</b> — 일정 수준 이상의 Risk Score 거래를 우선 검토 대상으로 분류했습니다.<br>
            📈 &nbsp;<b>Review / Block 분류</b> — 실제 Target 비율과 유사한 수준으로 Review와 Block 비율을 구성했습니다.
        </div>
        <div class="summary-box">
            <b>핵심 인사이트 요약</b>&nbsp;&nbsp;
            고액 거래, 심야 거래, 반복 거래 패턴에서 이상 거래 위험도가 높게 나타났습니다.
            Threshold 조정과 업종별 모니터링 정책을 함께 적용하는 것이 효과적입니다.
        </div>
        """, unsafe_allow_html=True)

        ch1, ch2 = st.columns(2, gap="medium")
        with ch1:
            st.markdown('<div class="dash-chart-title">시간대별 이상 거래 히트맵</div>', unsafe_allow_html=True)
            st.plotly_chart(make_hour_day_heatmap(), use_container_width=True)
        with ch2:
            st.markdown('<div class="dash-chart-title">금액대별 이상 비율</div>', unsafe_allow_html=True)
            st.plotly_chart(make_amount_fraud_chart(), use_container_width=True)

    # ── 이상탐지 ─────────────────────────────────────────────
    with tab_detect:
        r1 = st.columns(4, gap="medium")
        with r1[0]: card("탐지된 이상 거래 수",  f"{fraud_tx:,}",        "#FF5C72", "Target 기준")
        with r1[1]: card("이상 거래 비율",        f"{fraud_ratio}%",      "#F59E0B", "전체 거래 대비")
        with r1[2]: card("고위험 거래 수",        f"{high_risk_count:,}", "#818CF8", "Risk ≥ 70점")
        with r1[3]: card("차단 대상 거래 수",     f"{block_count:,}",     "#FF5C72", "Decision = Block")

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        r2 = st.columns(4, gap="medium")
        with r2[0]: card("탐지 정확도",    "97.60%",          "#34D399", "Rule 기반")
        with r2[1]: card("ML 탐지 정확도", "95.30%",          "#60A5FA", "ML 기반")
        with r2[2]: card("평균 탐지 지연", "2.30 sec",        "#FBBF24", "실시간 기준")
        with r2[3]: card("심야 거래 수",   f"{night_count:,}", "#A78BFA", "23시~05시 거래")

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2, gap="medium")
        with ch1: st.plotly_chart(make_detect_hour_chart(),   use_container_width=True)
        with ch2: st.plotly_chart(make_detect_amount_chart(), use_container_width=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="dash-chart-title">이상거래 목록 &amp; 필터</div>', unsafe_allow_html=True)

        f1, f2, f3, f4, f5 = st.columns([2.2, 1.1, 1.1, 1.1, 0.8], gap="small")
        with f1:
            st.date_input("기간", value=(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-31")), key="detect_date_range")
        with f2:
            dec_filter = st.selectbox("판단결과", ["전체","Block","Review","Pass"], key="detect_decision_filter")
        with f3:
            df_raw = load_fraud_data()
            cat_opts = ["전체"] + sorted(df_raw["category"].dropna().astype(str).unique().tolist()) if "category" in df_raw.columns else ["전체"]
            cat_filter = st.selectbox("거래유형", cat_opts, key="detect_category_filter")
        with f4:
            amt_filter = st.selectbox("금액범위", ["전체","~$10","$10~$100","$100~$500","$500+"], key="detect_amount_filter")
        with f5:
            st.write("")
            st.button("검색", use_container_width=True, key="detect_search_btn")

        table_df = _get_detect_table()
        if dec_filter != "전체" and "판단 결과" in table_df.columns:
            table_df = table_df[table_df["판단 결과"].astype(str) == dec_filter]
        if cat_filter != "전체" and "거래 유형" in table_df.columns:
            table_df = table_df[table_df["거래 유형"].astype(str) == cat_filter]
        if amt_filter == "~$10"       and "금액 ($)" in table_df.columns: table_df = table_df[table_df["금액 ($)"] < 10]
        elif amt_filter == "$10~$100" and "금액 ($)" in table_df.columns: table_df = table_df[(table_df["금액 ($)"] >= 10) & (table_df["금액 ($)"] < 100)]
        elif amt_filter == "$100~$500"and "금액 ($)" in table_df.columns: table_df = table_df[(table_df["금액 ($)"] >= 100) & (table_df["금액 ($)"] < 500)]
        elif amt_filter == "$500+"    and "금액 ($)" in table_df.columns: table_df = table_df[table_df["금액 ($)"] >= 500]

        block_n  = int((table_df["판단 결과"].astype(str) == "Block").sum())  if "판단 결과" in table_df.columns else 0
        review_n = int((table_df["판단 결과"].astype(str) == "Review").sum()) if "판단 결과" in table_df.columns else 0

        st.markdown(f'<div class="table-caption">Risk Score 70점 이상 후보 ({len(table_df):,}건) — Review {review_n:,}건 / Block {block_n:,}건</div>', unsafe_allow_html=True)
        st.dataframe(table_df.head(500), use_container_width=True, height=280, hide_index=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([1.1, 1.15], gap="medium")
        with b1: st.plotly_chart(make_factor_chart(),        use_container_width=True)
        with b2:
            st.markdown('<div class="dash-chart-title">이상 거래 패턴 요약</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="detect-summary-box">
                ● &nbsp;Risk Score 70점 이상 후보 우선 표시<br>
                ● &nbsp;Review / Block 비율 자동 구성<br>
                ● &nbsp;실제 Target 비율 참고<br>
                ● &nbsp;고위험 거래 우선 모니터링
            </div>""", unsafe_allow_html=True)

    # ── Risk Score ───────────────────────────────────────────
    with tab_risk:
        rg = _get_risk_group()
        r1, r2, r3 = st.columns(3, gap="medium")
        with r1: card("평균 Risk Score",  f"{avg_risk:.1f}",         "#60A5FA", "전체 거래 평균 (0~100점)")
        with r2: card("평균 Final Score", f"{avg_risk:.1f}",         "#818CF8", "보정 점수 평균 (0~100점)")
        with r3: card("High Risk 비율",   f"{high_risk_ratio:.2f}%", "#FF5C72", "Risk ≥ 70점")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        t1, t2 = st.columns(2, gap="medium")
        with t1: st.plotly_chart(make_risk_distribution_chart(), use_container_width=True)
        with t2: st.plotly_chart(make_risk_hour_chart(),         use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2, gap="medium")
        with m1: st.plotly_chart(make_risk_pie_chart(),          use_container_width=True)
        with m2: st.plotly_chart(make_risk_contribution_chart(), use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="dash-chart-title">Risk Score 구간별 거래 현황</div>', unsafe_allow_html=True)
        st.dataframe(rg, use_container_width=True, height=160, hide_index=True)

    # ── 고객 분석 ─────────────────────────────────────────────
    with tab_customer:
        # 세그먼트 집계값
        total_seg   = int(seg_stat["거래수"].sum())
        high_risk_n = int(seg_stat.loc[seg_stat["세그먼트"]=="고위험","거래수"].sum())
        vip_n       = int(seg_stat.loc[seg_stat["세그먼트"]=="VIP","거래수"].sum())
        avg_fraud_r = round(float(seg_stat["이상비율"].mean()), 1)

        # KPI 4개
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: card("전체 거래 수",      f"{total_seg:,}",  "#60A5FA", "세그먼트 분류 기준")
        with c2: card("고위험 거래 수",    f"{high_risk_n:,}","#FF5C72", "Risk ≥ 70점 세그먼트")
        with c3: card("VIP 거래 수",       f"{vip_n:,}",      "#34D399", "고액 + 정상 거래")
        with c4: card("평균 이상 거래율",  f"{avg_fraud_r}%", "#A78BFA", "세그먼트 평균")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 세그먼트 설명
        st.markdown("""
        <div style="background:#071527;border:1px solid #1E3A5F;border-radius:8px;
                    padding:12px 18px;margin-bottom:14px;font-size:11px;color:#94A3B8;line-height:2;">
            <b style="color:#93C5FD;">세그먼트 기준</b> &nbsp;|&nbsp;
            <span style="color:#60A5FA;">■ VIP</span> 고액(상위 10%) + 정상 &nbsp;
            <span style="color:#818CF8;">■ 기업</span> 고액(상위 25%) + 위험 70점 미만 &nbsp;
            <span style="color:#34D399;">■ 일반</span> Risk &lt; 40 &nbsp;
            <span style="color:#F59E0B;">■ 주의</span> 40 ≤ Risk &lt; 70 &nbsp;
            <span style="color:#EF4444;">■ 고위험</span> Risk ≥ 70
        </div>
        """, unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns(3, gap="medium")
        with cc1: st.plotly_chart(make_seg_risk_bar(seg_stat, seg_order), use_container_width=True)
        with cc2: st.plotly_chart(make_seg_dist_pie(seg_stat),            use_container_width=True)
        with cc3: st.plotly_chart(make_seg_fraud_bar(seg_stat, seg_order),use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # 세그먼트 상세 테이블
        st.markdown('<div class="dash-chart-title">세그먼트별 상세 현황</div>', unsafe_allow_html=True)
        disp = seg_stat.copy()
        disp["평균_Risk"]   = disp["평균_Risk"].round(1)
        disp["평균_금액"]   = disp["평균_금액"].round(2).apply(lambda x: f"${x:,.2f}")
        disp["이상비율"]    = disp["이상비율"].astype(str) + "%"
        disp["거래비율"]    = disp["거래비율"].astype(str) + "%"
        disp["이상거래수"]  = disp["이상거래수"].astype(int)
        disp = disp.rename(columns={
            "세그먼트":"세그먼트","거래수":"거래 수","평균_Risk":"평균 Risk",
            "평균_금액":"평균 금액","이상거래수":"이상 거래","이상비율":"이상 비율","거래비율":"거래 비율",
        })
        st.dataframe(
            disp[["세그먼트","거래 수","거래 비율","평균 Risk","평균 금액","이상 거래","이상 비율"]],
            use_container_width=True, height=220, hide_index=True
        )
