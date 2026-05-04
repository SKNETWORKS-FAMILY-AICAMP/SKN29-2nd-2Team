import streamlit as st
from pathlib import Path

from pages.main_page import show_main_page
from pages.dashboard_page import show_dashboard_page
from pages.model_compare_page import show_model_compare_page
from pages.simulation_page import show_simulation_page
from pages.tech_page import show_tech_page


def load_env_values() -> dict:
    env_path = Path(".env")
    values = {}

    if not env_path.exists():
        return values

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


ENV = load_env_values()
ADMIN_ID = ENV.get("ADMIN_ID")
ADMIN_PASSWORD = ENV.get("ADMIN_PASSWORD")

if not ADMIN_ID or not ADMIN_PASSWORD:
    raise ValueError("관리자 계정 환경변수가 설정되지 않았습니다. .env에 ADMIN_ID와 ADMIN_PASSWORD를 설정해주세요.")


st.set_page_config(
    page_title="FinGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# Global Style
# =========================
@st.cache_data(show_spinner=False)
def load_css() -> str:
    with open("style.css", "r", encoding="utf-8") as f:
        return f.read()


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0.6rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    .login-page {
        min-height: calc(100vh - 80px);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 32px 0;
    }

    .login-card {
        width: min(460px, 92vw);
        background: linear-gradient(180deg, rgba(9,20,40,0.98), rgba(6,15,30,0.99));
        border: 1px solid rgba(120,163,230,0.22);
        border-radius: 16px;
        box-shadow: 0 24px 60px rgba(1,6,18,0.38), inset 0 1px 0 rgba(255,255,255,0.04);
        padding: 34px 34px 26px;
        box-sizing: border-box;
    }

    .login-logo {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 8px;
    }

    .login-subtitle {
        color: #9FC4FF;
        font-size: 13px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
    }

    .login-desc {
        color: #A8B8D6;
        font-size: 13px;
        line-height: 1.7;
        text-align: center;
        margin-bottom: 24px;
    }

    .nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #081A30;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0 22px;
        height: 58px;
        margin-bottom: 8px;
        box-sizing: border-box;
    }

    .nav-header.brand-only {
        justify-content: flex-start;
    }

    .nav-brand {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 0.02em;
    }

    .nav-live {
        color: #22C55E;
        background: #082B22;
        border: 1px solid rgba(34,197,94,0.45);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 900;
        font-size: 12px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .nav-status-slot {
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #081A30;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        box-sizing: border-box;
        margin-bottom: 8px;
    }

    .nav-live::before {
        content: "●";
        font-size: 8px;
        animation: blink 1.4s ease-in-out infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .logout-slot .stButton > button {
        height: 34px;
        margin-top: 12px;
        border-radius: 20px;
        background: #082B22;
        border: 1px solid rgba(34,197,94,0.45);
        color: #22C55E;
        font-size: 12px;
        font-weight: 900;
        padding: 6px 14px;
    }

    .logout-slot .stButton > button:hover {
        background: #0E3A2E;
        border-color: rgba(74,222,128,0.75);
        color: #4ADE80;
    }

    .stButton > button {
        background: #071527;
        color: #F8FAFC;
        border: 1px solid #2563EB;
        border-radius: 6px;
        font-weight: 900;
        height: 36px;
        width: 100%;
        transition: background 0.1s, color 0.1s, border 0.1s !important;
    }

    .stButton > button:hover {
        background: #0B2A55;
        color: #FFFFFF;
        border-color: #60A5FA;
    }

    .active-page {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
    }

    .nav-active > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
    }

    .login-panel {
        max-width: 420px;
        margin: 34px auto 0 auto;
        background: #071527;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 28px 30px 26px;
        box-sizing: border-box;
    }

    .login-title {
        color: #F8FAFC;
        font-size: 24px;
        font-weight: 900;
        margin-bottom: 8px;
        text-align: center;
    }

    .login-desc {
        color: #94A3B8;
        font-size: 13px;
        line-height: 1.7;
        text-align: center;
        margin-bottom: 22px;
    }

    .login-form-scope [data-testid="stTextInput"] label {
        color: #CBD5E1 !important;
        font-size: 12px;
        font-weight: 900;
    }

    .login-form-scope [data-testid="stTextInput"] input {
        background: #0B1F38;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        color: #F8FAFC !important;
        height: 42px;
    }

    .login-form-scope .stButton > button {
        background: #2563EB;
        border: 1px solid #2563EB;
        border-radius: 8px;
        color: #FFFFFF;
        height: 42px;
        font-size: 13px;
        font-weight: 900;
    }

    .login-form-scope .stButton > button:hover {
        background: #1D4ED8;
        border-color: #60A5FA;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stDateInput"] label {
        color: #CBD5E1 !important;
        font-weight: 800 !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {
        background: #0B1F38 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(120,163,230,0.28) !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] > div {
        background: #0B1F38 !important;
        border-color: rgba(120,163,230,0.32) !important;
        color: #F8FAFC !important;
    }

    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #F8FAFC !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: #071527 !important;
        color: #F8FAFC !important;
    }

    [role="listbox"],
    [role="option"] {
        background: #071527 !important;
        color: #F8FAFC !important;
    }

    [role="option"]:hover {
        background: #0B2A55 !important;
        color: #FFFFFF !important;
    }

    .st-key-login_form_scope {
        width: min(460px, 92vw);
        margin: 10vh auto 0;
        padding: 34px;
        background: linear-gradient(180deg, rgba(9,20,40,0.98), rgba(6,15,30,0.99));
        border: 1px solid rgba(120,163,230,0.22);
        border-radius: 16px;
        box-shadow: 0 24px 60px rgba(1,6,18,0.38), inset 0 1px 0 rgba(255,255,255,0.04);
        box-sizing: border-box;
    }

    .st-key-login_form_scope [data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    .st-key-login_form_scope .stButton > button {
        height: 44px;
        border-radius: 10px;
        background: #2563EB;
        border: 1px solid #60A5FA;
        color: #FFFFFF;
        font-size: 13px;
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Page/Auth State
# =========================
MENUS = ["메인", "대시보드", "모델 비교", "시뮬레이션", "기술 소개"]
ADMIN_PAGES = ["대시보드", "모델 비교", "시뮬레이션"]
PAGE_IDS = {
    "메인": "main",
    "대시보드": "dashboard",
    "모델 비교": "modelcompare",
    "시뮬레이션": "simulation",
    "기술 소개": "tech",
}

if "page" not in st.session_state:
    st.session_state["page"] = "메인"

if "page_version" not in st.session_state:
    st.session_state["page_version"] = 0

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False


# =========================
# Top Navigation
# =========================
def nav():
    if st.session_state["is_admin"]:
        brand_col, status_col, logout_col = st.columns([7.4, 1.7, 1.25], gap="small")
        with brand_col:
            st.markdown(
                """
                <div class="nav-header brand-only">
                    <span class="nav-brand">🛡 FinGuard</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with status_col:
            st.markdown(
                '<div class="nav-status-slot"><span class="nav-live">SYSTEM ONLINE</span></div>',
                unsafe_allow_html=True,
            )
        with logout_col:
            st.markdown('<div class="logout-slot">', unsafe_allow_html=True)
            if st.button("LOGOUT", key="admin_logout_btn", use_container_width=True):
                st.session_state["is_admin"] = False
                if st.session_state["page"] in ADMIN_PAGES:
                    st.session_state["page"] = "메인"
                st.session_state["page_version"] += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="nav-header">
                <span class="nav-brand">🛡 FinGuard</span>
                <span class="nav-live">SYSTEM ONLINE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_cols = st.columns(len(MENUS), gap="small")

    for col, menu in zip(nav_cols, MENUS):
        with col:
            if st.session_state["page"] == menu:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            if st.button(menu, key=f"nav_{menu}", use_container_width=True):
                if st.session_state["page"] != menu:
                    st.session_state["page"] = menu
                    st.session_state["page_version"] += 1
                    st.rerun()
            if st.session_state["page"] == menu:
                st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Login Gate
# =========================
def show_login_page():
    with st.container(key="login_form_scope"):
        st.markdown(
            """
            <div class="login-logo">🛡 FinGuard</div>
            <div class="login-subtitle">관리자 대시보드</div>
            <div class="login-desc">
                이상거래 탐지 결과와 모델 운영 지표를 확인하는 관리자 전용 화면입니다.<br>
                관리자 계정으로 로그인해주세요.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="login-form-scope">', unsafe_allow_html=True)
        with st.form("admin_login_form"):
            admin_id = st.text_input("ID", placeholder="관리자 ID")
            admin_pw = st.text_input("Password", type="password", placeholder="관리자 비밀번호")
            submitted = st.form_submit_button("로그인", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if admin_id == ADMIN_ID and admin_pw == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.session_state["page_version"] += 1
            st.success("로그인되었습니다.")
            st.rerun()
        else:
            st.error("ID 또는 비밀번호가 올바르지 않습니다.")


if not st.session_state["is_admin"]:
    show_login_page()
    st.stop()


nav()


current_page = st.session_state["page"]
current_page_id = PAGE_IDS.get(current_page, "main")
current_page_version = st.session_state["page_version"]
current_page_key = f"pageroot{current_page_id}{current_page_version}"

# Streamlit이 이전 페이지 DOM을 남겨도 현재 페이지 루트만 노출한다.
st.markdown(
    f"""
    <style>
    [class*="st-key-page_root_"],
    [class*="st-key-page-root-"],
    [class*="st-key-pageroot"] {{
        display: none !important;
    }}

    .st-key-{current_page_key} {{
        display: block !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 관리자 페이지는 로그인 전에는 함수 호출 자체를 막는다.
with st.container(key=current_page_key):
    if current_page == "메인":
        show_main_page()

    elif current_page == "대시보드":
        show_dashboard_page()

    elif current_page == "모델 비교":
        show_model_compare_page()

    elif current_page == "시뮬레이션":
        show_simulation_page()

    elif current_page == "기술 소개":
        show_tech_page()

    else:
        st.session_state["page"] = "메인"
        st.rerun()
