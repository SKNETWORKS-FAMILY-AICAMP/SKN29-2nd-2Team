import streamlit as st
import plotly.graph_objects as go
from data.data_loader import load_fraud_data

ARROW_DIV  = '<div style="position:absolute;right:-14px;top:50%;transform:translateY(-50%);color:#2563EB;font-size:22px;font-weight:900;z-index:1;">›</div>'
ARROW_PIPE = '<div style="position:absolute;right:-12px;top:50%;transform:translateY(-50%);color:#2563EB;font-size:20px;font-weight:900;z-index:1;">›</div>'


def show_tech_page():
    df = load_fraud_data()

    total_tx   = len(df)
    categories = df["category"].nunique()
    states     = df["state"].nunique() if "state" in df.columns else 51

    st.markdown('<h1 class="page-title">기술 소개</h1>', unsafe_allow_html=True)

    # ── 상단 소개 ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:24px 0 32px;">
        <span style="color:#93C5FD;font-size:14px;font-weight:600;display:block;margin-bottom:12px;
                     letter-spacing:0.08em;">ML-based Fraud Detection System</span>
        <span style="color:#F8FAFC;font-size:24px;font-weight:900;display:block;line-height:1.5;margin-bottom:12px;">
            FinGuard는 실제 카드 거래 데이터를 기반으로<br>
            머신러닝 알고리즘을 활용해 이상 거래를 실시간으로 탐지합니다.
        </span>
        <span style="color:#94A3B8;font-size:15px;display:block;margin-bottom:16px;line-height:1.9;">
            Rule 기반 한계를 보완하고, 복합 점수 기반 의사결정을 통해<br>
            보다 정확한 탐지와 오탐 감소를 동시에 달성합니다.
        </span>
        <span style="color:#64748B;font-size:14px;display:block;">
            대규모 실제 거래 데이터 기반 &nbsp;·&nbsp; 업종 {categories}개 &nbsp;·&nbsp; 지역 {states}개 &nbsp;·&nbsp; 실시간 탐지 및 의사결정 지원
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 서비스 특장점 ─────────────────────────────────────────
    st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">서비스 특장점</p>',
                unsafe_allow_html=True)

    feats = [
        ("🧠", "룰 기반 한계 보완 → ML 탐지", "#60A5FA",
         "고정된 룰로 탐지하기 어려운 새로운 이상 패턴에 머신러닝이 적응하여 더 정확하게 탐지합니다."),
        ("🛡", "복합 위험 점수 (Final Score)", "#818CF8",
         "룰 기반 Risk Score, ML 예측 Score, 이상탐지 Score를 결합하여 더 엄격하고 신뢰도 높은 판단을 제공합니다."),
        ("⚡", "실시간 의사결정 지원", "#34D399",
         "최종 점수를 기반으로 거래를 즉시 평가하고, 차단 / 검토 / 정상 의사결정을 실시간으로 수행합니다."),
    ]

    for col, (icon, title, color, desc) in zip(st.columns(3, gap="medium"), feats):
        with col:
            st.markdown(
                f'<div style="background:#0D1F35;border:1px solid #1E3A5F;border-radius:16px;'
                f'padding:24px 22px;border-left:4px solid {color};height:170px;'
                f'box-sizing:border-box;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                f'<span style="font-size:28px;">{icon}</span>'
                f'<span style="color:#F8FAFC;font-size:15px;font-weight:800;">{title}</span>'
                f'</div>'
                f'<span style="color:#94A3B8;font-size:13px;line-height:1.7;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 시스템 아키텍처 ───────────────────────────────────────
    st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">시스템 아키텍처</p>',
                unsafe_allow_html=True)

    arch_steps = [
        ("🗄",  "입력 데이터",     "#60A5FA", "카드 거래 데이터\n(시간, 금액, 가맹점,\n거래 등)"),
        ("⚙️",  "피처 엔지니어링", "#818CF8", "시간, 금액, 거리, 업종 등\n다양한 피처 생성 및 가공"),
        ("🤖", "모델 연산",       "#F472B6", "Risk Score (룰 기반)\nML Score (XGBoost)\nAnomaly Score (AutoEncoder)"),
        ("📊", "최종 점수 계산",  "#FBBF24", "가중치 기반 점수 결합\n(Final Score)"),
        ("🎯", "의사결정",        "#34D399", "차단 (Block)\n검토 (Review)\n정상 (Pass)"),
    ]

    for i, (col, (icon, title, color, desc)) in enumerate(
            zip(st.columns(len(arch_steps), gap="small"), arch_steps)):
        arrow = ARROW_DIV if i < len(arch_steps) - 1 else ""
        with col:
            st.markdown(
                f'<div style="background:#0D1F35;border:1px solid {color}55;border-radius:14px;'
                f'padding:18px 10px;text-align:center;position:relative;border-top:3px solid {color};'
                f'height:220px;box-sizing:border-box;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;gap:8px;">'
                f'<span style="font-size:30px;">{icon}</span>'
                f'<span style="color:{color};font-size:13px;font-weight:800;">{title}</span>'
                f'<span style="color:#64748B;font-size:11px;line-height:1.7;white-space:pre-line;">{desc}</span>'
                f'{arrow}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 핵심 기술 스택 + Final Score 구성 (한 행 2열, 제목/상자 따로) ──
    stack_col, final_col = st.columns([1, 1.5], gap="medium")

    with stack_col:
        st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">핵심 기술 스택</p>',
                    unsafe_allow_html=True)
        stack_rows = [
            ("ML 모델",     "#60A5FA", "XGBoost"),
            ("이상탐지",    "#F472B6", "AutoEncoder"),
            ("데이터 처리", "#818CF8", "Pandas, NumPy"),
            ("웹/대시보드", "#34D399", "Streamlit"),
            ("기타",        "#FBBF24", "Scikit-learn, Joblib, Plotly, CSS/HTML"),
        ]
        for label, color, badge in stack_rows:
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:13px 18px;margin-bottom:7px;'
                f'background:#0D1F35;border:1px solid #1E3A5F;border-radius:10px;border-left:4px solid {color};">'
                f'<span style="color:#F8FAFC;font-size:14px;font-weight:700;min-width:110px;">{label}</span>'
                f'<span style="color:{color};font-size:14px;font-weight:600;">{badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 오른쪽: Final Score
    with final_col:
        st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">Final Score 구성</p>',
                    unsafe_allow_html=True)
        fs_left, fs_right = st.columns([1, 1.3], gap="small")

        with fs_left:
            fig = go.Figure(go.Pie(
                labels=["Risk Score (룰 기반)", "ML Score (XGBoost)", "Anomaly Score (AutoEncoder)"],
                values=[40, 40, 20],
                hole=0.6,
                marker=dict(colors=["#60A5FA", "#818CF8", "#34D399"]),
                textinfo="percent",
                textfont=dict(size=14, color="white"),
            ))
            fig.add_annotation(
                text="Final<br>Score", x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color="#F8FAFC", family="Arial Black"),
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0A1A2F",
                height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with fs_right:
            for color, pct, name, desc in [
                ("#60A5FA", "40%", "Risk Score (룰 기반)",        "거래 금액, 시간, 거리 등 규칙 기반 위험 점수"),
                ("#818CF8", "40%", "ML Score (XGBoost)",          "머신러닝 모델의 이상거래 예측 점수"),
                ("#34D399", "20%", "Anomaly Score (AutoEncoder)", "비지도 학습 기반 이상 탐지 점수"),
            ]:
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 14px;'
                    f'margin-bottom:7px;background:#0D1F35;border:1px solid #1E3A5F;border-radius:10px;">'
                    f'<div style="width:10px;height:10px;border-radius:50%;background:{color};'
                    f'margin-top:4px;flex-shrink:0;"></div>'
                    f'<div>'
                    f'<span style="color:{color};font-size:15px;font-weight:900;">{pct}&nbsp;</span>'
                    f'<span style="color:#F8FAFC;font-size:13px;font-weight:700;">{name}</span><br>'
                    f'<span style="color:#64748B;font-size:12px;">{desc}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div style="background:#1E3A5F;border-radius:10px;padding:11px 14px;margin-top:4px;">'
                '<span style="color:#93C5FD;font-size:13px;font-weight:700;">가중치 예시</span><br>'
                '<span style="color:#CBD5E1;font-size:13px;">Risk 40% + ML 40% + Anomaly 20% = Final Score</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 데이터 파이프라인 ─────────────────────────────────────
    st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">데이터 파이프라인</p>',
                unsafe_allow_html=True)

    pipe_steps = [
        ("📥", "데이터 수집",      "#60A5FA", "실시간 카드 거래\n데이터 수집"),
        ("🔄", "전처리",           "#818CF8", "결측치 처리, 이상치\n처리, 데이터 정제"),
        ("⚙️",  "피처 엔지니어링", "#F472B6", "시간, 금액, 거리,\n업종 등 110+ 피처 생성"),
        ("🤖", "모델 예측",        "#FBBF24", "Risk / ML / Anomaly\n모델 예측 수행"),
        ("📊", "점수 결합",        "#34D399", "가중치 기반 점수\n결합 (Final Score)"),
        ("🎯", "의사결정 & 알림",  "#F87171", "차단 / 검토 / 정상\n판정 및 알림 전송"),
    ]

    for i, (col, (icon, title, color, desc)) in enumerate(
            zip(st.columns(len(pipe_steps), gap="small"), pipe_steps)):
        arrow = ARROW_PIPE if i < len(pipe_steps) - 1 else ""
        with col:
            st.markdown(
                f'<div style="background:#0D1F35;border:1px solid {color}55;border-radius:12px;'
                f'padding:16px 8px;text-align:center;border-top:3px solid {color};position:relative;'
                f'height:170px;box-sizing:border-box;display:flex;flex-direction:column;'
                f'justify-content:center;gap:6px;">'
                f'<span style="font-size:26px;">{icon}</span>'
                f'<span style="color:{color};font-size:12px;font-weight:800;">{title}</span>'
                f'<span style="color:#64748B;font-size:11px;line-height:1.6;white-space:pre-line;">{desc}</span>'
                f'{arrow}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 기대 효과 ─────────────────────────────────────────────
    st.markdown('<p style="color:#93C5FD;font-size:17px;font-weight:700;margin-bottom:14px;">기대 효과</p>',
                unsafe_allow_html=True)

    effects = [
        ("🎯", "탐지 성능 향상", "#60A5FA", "Recall 93.7% 달성",    "룰 기반 이상거래 탐지 및\nRecall 향상"),
        ("📉", "오탐 감소",      "#34D399", "Precision 91.7% 달성", "불필요한 알림 감소로\n운영 효율성 개선"),
        ("⚡", "실시간 처리",    "#FBBF24", "밀리세컨드 단위 예측", "거래 발생 즉시\n의사결정 지원"),
        ("🛡️", "안정성 강화",   "#818CF8", "다중 모델 검증",        "다중 모델 결합으로\n판정 신뢰성 향상"),
    ]

    for col, (icon, title, color, badge, desc) in zip(st.columns(4, gap="medium"), effects):
        with col:
            st.markdown(
                f'<div style="background:#0D1F35;border:1px solid #1E3A5F;border-radius:16px;'
                f'padding:22px 18px;border-top:4px solid {color};height:210px;'
                f'box-sizing:border-box;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div>'
                f'<span style="font-size:30px;display:block;margin-bottom:8px;">{icon}</span>'
                f'<span style="color:#F8FAFC;font-size:15px;font-weight:800;display:block;margin-bottom:6px;">{title}</span>'
                f'<span style="color:#94A3B8;font-size:13px;line-height:1.6;display:block;white-space:pre-line;">{desc}</span>'
                f'</div>'
                f'<span style="background:{color}22;color:{color};border:1px solid {color};'
                f'border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700;'
                f'display:inline-block;margin-top:8px;">{badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
