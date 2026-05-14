import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import text


def _prepare_fraud_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV/DB 어느 쪽에서 읽어도 대시보드가 기대하는 컬럼을 맞춘다.
    """
    df = df.copy()

    if "cc_num" not in df.columns:
        df["cc_num"] = [f"****-****-****-{1000 + i}" for i in range(len(df))]
    else:
        fallback_cards = pd.Series(
            [f"****-****-****-{1000 + i}" for i in range(len(df))],
            index=df.index,
        )
        df["cc_num"] = df["cc_num"].fillna(fallback_cards).astype(str)

    if "category" not in df.columns:
        df["category"] = "Unknown"
    else:
        df["category"] = df["category"].fillna("Unknown")

    if "state" not in df.columns:
        df["state"] = "Unknown"

    if "distance_km" not in df.columns:
        rng = np.random.default_rng(42)
        df["distance_km"] = rng.uniform(1, 120, len(df))

    if "time_since_last_min" not in df.columns:
        rng = np.random.default_rng(42)
        df["time_since_last_min"] = rng.integers(1, 240, len(df))

    if "amt_zscore" not in df.columns:
        df["amt_zscore"] = (df["amt"] - df["amt"].mean()) / df["amt"].std()

    if "dist_zscore" not in df.columns:
        df["dist_zscore"] = (
            (df["distance_km"] - df["distance_km"].mean())
            / df["distance_km"].std()
        )

    if "hour_dev" not in df.columns:
        df["hour_dev"] = (
            abs(df["trans_hour"] - df["trans_hour"].mean())
            / df["trans_hour"].std()
        )

    if "is_night" not in df.columns:
        df["is_night"] = (
            (df["trans_hour"] >= 23) | (df["trans_hour"] <= 5)
        ).astype(int)

    if "high_amt_far" not in df.columns:
        df["high_amt_far"] = (
            (df["amt_zscore"] > 2) & (df["dist_zscore"] > 2)
        ).astype(int)
    else:
        df["high_amt_far"] = df["high_amt_far"].fillna(False).astype(int)

    amt_score = (
        (df["amt_zscore"].clip(lower=0.5, upper=3.5) - 0.5) / 3.0 * 40
    ).clip(lower=0)

    high_amt_far_score = df["high_amt_far"] * 20

    hour_cap = df["hour_dev"].quantile(0.90)
    hour_score = (
        df["hour_dev"].clip(lower=0, upper=hour_cap) / hour_cap * 20
    ) if hour_cap else 0

    night_score = df["is_night"] * 12

    dist_cap = df["distance_km"].quantile(0.85)
    dist_score = (
        df["distance_km"].clip(lower=0, upper=dist_cap) / dist_cap * 8
    ) if dist_cap else 0

    df["risk_score"] = (
        amt_score
        + high_amt_far_score
        + hour_score
        + night_score
        + dist_score
    ).clip(0, 100).round(1)

    df["decision"] = pd.cut(
        df["risk_score"],
        bins=[-1, 49.9, 69.9, 100],
        labels=["Pass", "Review", "Block"]
    )

    return df


def _load_fraud_data_from_db(data_split: str = "train") -> pd.DataFrame:
    """
    MySQL transaction_features를 기준으로 앱에서 쓰는 거래 DataFrame을 구성한다.
    raw_transactions는 화면 표시용 카드/가맹점/일시 정보를 보강하는 용도다.
    """
    from db.database import engine

    query = text(
        """
        SELECT
            tf.txn_id,
            rt.cc_num,
            rt.merchant,
            rt.trans_date_trans_time,
            COALESCE(tf.category, rt.category) AS category,
            tf.amt,
            tf.gender,
            tf.state,
            tf.zip_code,
            tf.customer_lat,
            tf.customer_long,
            tf.city_pop,
            tf.unix_time,
            tf.merchant_lat,
            tf.merchant_long,
            tf.is_fraud,
            tf.trans_hour,
            tf.trans_dayofweek,
            tf.trans_month,
            tf.trans_day,
            tf.age,
            tf.distance_km,
            tf.amt_zscore,
            tf.hour_dev,
            tf.high_amt_far
        FROM transaction_features AS tf
        LEFT JOIN raw_transactions AS rt
            ON rt.txn_id = tf.txn_id
        WHERE tf.data_split = :data_split
        ORDER BY tf.txn_id
        """
    )

    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"data_split": data_split})


def _load_fraud_data_from_csv() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "cc_fraud_train_processed.csv"
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner="데이터 로딩 중...")
def load_fraud_data():
    """
    DB 데이터를 우선 읽고, 실패하거나 비어 있으면 CSV로 fallback한다.

    [Risk Score 설계]
    목표: Block 8,000~12,000건 / Review 15,000~20,000건

    [피처별 최대 기여점수]
    1. amt_zscore  : zscore 0.5~3.5 선형  → 최대 40점  (넓은 구간으로 더 많은 거래 포함)
    2. high_amt_far: 고액+장거리 동시     → 20점 고정
    3. hour_dev    : 상위 10% 기준 선형   → 최대 20점  (기준 완화 → 더 많이 적용)
    4. is_night    : 심야 거래            → 12점 고정
    5. distance_km : 상위 15% 기준 선형   → 최대 8점   (기준 완화 → 더 많이 적용)
    → 합산 최대 100점

    [Decision 기준]
    Pass   :  0 ~ 49  (정상)
    Review : 50 ~ 69  (주의)
    Block  : 70 ~ 100 (차단)
    """
    try:
        df = _load_fraud_data_from_db(data_split="train")
        if df.empty:
            raise ValueError("transaction_features 테이블에 train 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"DB 연결/조회 실패로 CSV 데이터를 사용합니다. ({e})")
        df = _load_fraud_data_from_csv()

    return _prepare_fraud_frame(df)


@st.cache_data(show_spinner=False, ttl=3600)
def load_kpi_summary() -> dict:
    """대시보드 KPI 집계만 따로 캐싱 — 매 rerun 집계 재계산 방지"""
    df = load_fraud_data()
    total  = len(df)
    fraud  = int(df["is_fraud"].sum())
    high   = int((df["risk_score"] >= 70).sum())
    block  = int((df["decision"] == "Block").sum())
    review = int((df["decision"] == "Review").sum())
    night  = int(df["is_night"].sum()) if "is_night" in df.columns else 0
    return dict(
        total_tx=total,
        fraud_tx=fraud,
        fraud_ratio=round(fraud / total * 100, 2) if total else 0,
        avg_risk=round(float(df["risk_score"].mean()), 1),
        high_risk_count=high,
        high_risk_ratio=round(high / total * 100, 2) if total else 0,
        block_count=block,
        review_count=review,
        night_count=night,
        customer_count=df["cc_num"].nunique() if "cc_num" in df.columns else total,
    )
