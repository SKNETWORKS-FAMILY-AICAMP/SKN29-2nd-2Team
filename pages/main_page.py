import streamlit as st
from data.data_loader import load_fraud_data


def show_main_page():
    df = load_fraud_data()

    total_tx    = len(df)
    fraud_tx    = int(df["is_fraud"].sum())
    fraud_ratio = round((fraud_tx / total_tx) * 100, 2)
    avg_amt     = round(df["amt"].mean(), 2)

    top_category = (
        df[df["is_fraud"] == 1]["category"]
        .value_counts()
        .idxmax()
    )

    # ── Hero 섹션 ─────────────────────────────────────────────
    st.markdown("""
    <section class="hero">
        <div>
            <p class="eyebrow">FinGuard Fraud Detection System</p>
            <h1>ML 기반 카드 거래<br>이상 탐지 시스템</h1>
            <p class="hero-desc">
                실제 카드 거래 데이터를 기반으로 이상 거래 패턴을 탐지하고<br>
                금융 사고를 사전에 차단하는 머신러닝 시스템입니다.
            </p>
            <div class="hero-btn">실시간 모니터링 운영중</div>
        </div>
        <div class="shield-box">
            <div class="shield">🛡</div>
        </div>
    </section>
    """, unsafe_allow_html=True)

    # ── 동적 Alert ────────────────────────────────────────────
    if fraud_ratio > 0.5:
        st.markdown(f"""
        <style>
        @keyframes mainAlertBlink {{
            0%   {{ opacity: 1;    box-shadow: 0 0 24px rgba(239,68,68,0.9); }}
            50%  {{ opacity: 0.45; box-shadow: 0 0 6px  rgba(239,68,68,0.2); }}
            100% {{ opacity: 1;    box-shadow: 0 0 24px rgba(239,68,68,0.9); }}
        }}
        .main-fraud-alert {{
            animation: mainAlertBlink 1.2s ease-in-out infinite;
            background: linear-gradient(90deg, #450a0a 0%, #7f1d1d 100%);
            border: 2px solid #EF4444;
            border-radius: 14px;
            padding: 18px 26px;
            margin-top: 22px;
            display: flex;
            align-items: center;
            gap: 18px;
        }}
        .main-fraud-alert .al-icon  {{ font-size: 30px; flex-shrink: 0; }}
        .main-fraud-alert .al-body  {{ flex: 1; }}
        .main-fraud-alert .al-title {{
            color: #FCA5A5; font-size: 15px; font-weight: 800;
            display: block; margin-bottom: 5px;
        }}
        .main-fraud-alert .al-desc  {{
            color: #FECACA; font-size: 13px; font-weight: 600; display: block;
        }}
        .main-fraud-alert .al-badge {{
            background: #DC2626; color: #FFFFFF;
            font-size: 22px; font-weight: 900;
            padding: 10px 20px; border-radius: 10px;
            white-space: nowrap; flex-shrink: 0;
        }}
        </style>
        <div class="main-fraud-alert">
            <span class="al-icon">🚨</span>
            <div class="al-body">
                <span class="al-title">⚠️ FRAUD ALERT — 이상 거래 비율 임계치 초과 (기준: 0.5%)</span>
                <span class="al-desc">
                    총 {total_tx:,}건 중 {fraud_tx:,}건 이상 거래 감지 —
                    즉각적인 모니터링 및 조치가 필요합니다.
                </span>
            </div>
            <span class="al-badge">{fraud_ratio}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 간격 + 섹션 타이틀 ───────────────────────────────────
    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">오늘의 요약</h3>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── KPI 카드 — main 전용 클래스명 사용 ──────────────────
    st.markdown("""
    <style>
    .main-kpi-card {
        background: linear-gradient(135deg, #0f2744 0%, #0a1a2f 100%);
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 22px 20px 18px;
        box-sizing: border-box;
        min-height: 110px;
    }
    </style>
    """, unsafe_allow_html=True)

    kpis = [
        ("📊", "총 거래 건수",   f"{total_tx:,}",    "전체 데이터 기준",                "#60A5FA"),
        ("⚠️", "이상 거래 건수", f"{fraud_tx:,}",    "is_fraud = 1",                    "#F87171"),
        ("📈", "Fraud 비율",     f"{fraud_ratio}%",  "이상 거래 비중",                  "#FB923C"),
        ("💳", "평균 거래 금액", f"${avg_amt:,.2f}", f"Top Fraud 업종: {top_category}", "#34D399"),
    ]

    cols = st.columns(4, gap="medium")
    for col, (icon, label, value, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="main-kpi-card" style="border-left:4px solid {color};">'
                f'<span style="color:#93C5FD;font-size:13px;font-weight:600;display:block;margin-bottom:10px;">{icon} {label}</span>'
                f'<span style="color:{color};font-size:34px;font-weight:900;display:block;">{value}</span>'
                f'<span style="color:#64748B;font-size:12px;font-weight:600;display:block;margin-top:8px;">{sub}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
