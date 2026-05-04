

# db/models.py
# ERD 기준 ORM 테이블 정의
# 이상거래 탐지 시스템
# FastAPI + SQLAlchemy ORM + MySQL 기준

from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    TIMESTAMP,
    ForeignKey,
    Text,
    Boolean,
    text,
)

from sqlalchemy.orm import relationship

from db.database import Base


# ============================================================
# 1. 원본 거래 테이블
# ============================================================

class RawTransaction(Base):
    """
    raw_transactions

    역할:
    - Kaggle 원본 CSV 데이터를 최대한 원본에 가깝게 저장하는 테이블
    - 전처리/피처 엔지니어링 이전의 기준 데이터
    - 재전처리, 재학습, 추적 가능성을 위해 사용

    주의:
    - 파생 컬럼(trans_hour, age, distance_km 등)은 저장하지 않음
    - 파생 컬럼은 transaction_features 테이블에서 관리
    """

    __tablename__ = "raw_transactions"

    # 내부 거래 식별자
    txn_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="시스템 내부 거래 ID. 전체 거래의 기준 PK"
    )

    # 데이터 출처 구분
    data_split = Column(
        String(20),
        nullable=True,
        comment="데이터 구분값. 예: train, test"
    )

    data_source = Column(
        String(50),
        nullable=True,
        comment="데이터 출처. 예: kaggle"
    )

    # 원본 거래 시간
    trans_date_trans_time = Column(
        DateTime,
        nullable=True,
        comment="원본 거래 발생 일시"
    )

    # 카드 번호
    cc_num = Column(
        String(30),
        nullable=True,
        comment="카드 번호. 실제 서비스에서는 해시/마스킹 권장"
    )

    # 가맹점 정보
    merchant = Column(
        String(255),
        nullable=True,
        comment="가맹점명"
    )

    category = Column(
        String(100),
        nullable=True,
        comment="거래 카테고리"
    )

    # 거래 금액
    amt = Column(
        Float,
        nullable=True,
        comment="거래 금액"
    )

    # 사용자 이름
    first_name = Column(
        String(100),
        nullable=True,
        comment="사용자 이름"
    )

    last_name = Column(
        String(100),
        nullable=True,
        comment="사용자 성"
    )

    # 사용자 성별
    gender = Column(
        String(1),
        nullable=True,
        comment="사용자 성별"
    )

    # 사용자 주소 정보
    street = Column(
        String(255),
        nullable=True,
        comment="사용자 상세 주소"
    )

    city = Column(
        String(100),
        nullable=True,
        comment="사용자 거주 도시"
    )

    state = Column(
        String(20),
        nullable=True,
        comment="사용자 거주 주/지역"
    )

    zip_code = Column(
        String(20),
        nullable=True,
        comment="우편번호. 원본 zip 컬럼명 변경"
    )

    # 사용자 위치 정보
    customer_lat = Column(
        Float,
        nullable=True,
        comment="사용자 위도. 원본 lat 컬럼명 변경"
    )

    customer_long = Column(
        Float,
        nullable=True,
        comment="사용자 경도. 원본 long 컬럼명 변경"
    )

    city_pop = Column(
        Integer,
        nullable=True,
        comment="사용자 거주 도시 인구"
    )

    # 사용자 직업/생년월일
    job = Column(
        String(255),
        nullable=True,
        comment="사용자 직업"
    )

    dob = Column(
        Date,
        nullable=True,
        comment="사용자 생년월일"
    )

    # 원본 거래 고유번호
    trans_num = Column(
        String(255),
        nullable=True,
        unique=True,
        comment="원본 데이터의 거래 고유번호"
    )

    unix_time = Column(
        BigInteger,
        nullable=True,
        comment="거래 발생 시각의 Unix timestamp"
    )

    # 가맹점 위치 정보
    merchant_lat = Column(
        Float,
        nullable=True,
        comment="가맹점 위도. 원본 merch_lat 컬럼명 변경"
    )

    merchant_long = Column(
        Float,
        nullable=True,
        comment="가맹점 경도. 원본 merch_long 컬럼명 변경"
    )

    # 실제 사기 여부
    is_fraud = Column(
        Integer,
        nullable=True,
        comment="실제 이상거래 여부. 0: 정상, 1: 사기"
    )

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="DB 적재 시각"
    )

    # 관계 설정
    feature = relationship(
        "TransactionFeature",
        back_populates="raw_transaction",
        uselist=False,
        cascade="all, delete-orphan"
    )

    prediction = relationship(
        "ModelPrediction",
        back_populates="raw_transaction",
        uselist=False,
        cascade="all, delete-orphan"
    )

    anomaly_score = relationship(
        "AnomalyScore",
        back_populates="raw_transaction",
        uselist=False,
        cascade="all, delete-orphan"
    )

    final_score = relationship(
        "FinalScore",
        back_populates="raw_transaction",
        uselist=False,
        cascade="all, delete-orphan"
    )

    alerts = relationship(
        "Alert",
        back_populates="raw_transaction",
        cascade="all, delete-orphan"
    )


# ============================================================
# 2. 모델 입력용 최종 Feature 테이블
# ============================================================

class TransactionFeature(Base):
    """
    transaction_features

    역할:
    - 모델 입력에 사용할 최종 전처리/피처 엔지니어링 결과 저장
    - 거래 1건당 feature 1건
    - txn_id를 PK이자 FK로 사용

    관계:
    - raw_transactions 1 : 1 transaction_features
    """

    __tablename__ = "transaction_features"

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        primary_key=True,
        comment="거래 ID. raw_transactions.txn_id를 참조하는 PK/FK"
    )

    # 데이터 출처 구분
    data_split = Column(
        String(20),
        nullable=True,
        comment="데이터 구분값. 예: train, test"
    )

    data_source = Column(
        String(50),
        nullable=True,
        comment="데이터 출처. 예: kaggle"
    )

    category = Column(String(100), nullable=True, comment="거래 카테고리")
    amt = Column(Float, nullable=True, comment="거래 금액")
    gender = Column(String(1), nullable=True, comment="사용자 성별")
    state = Column(String(20), nullable=True, comment="사용자 거주 주/지역")
    zip_code = Column(String(20), nullable=True, comment="우편번호")

    customer_lat = Column(Float, nullable=True, comment="사용자 위도")
    customer_long = Column(Float, nullable=True, comment="사용자 경도")
    city_pop = Column(Integer, nullable=True, comment="사용자 거주 도시 인구")
    unix_time = Column(BigInteger, nullable=True, comment="거래 Unix timestamp")

    merchant_lat = Column(Float, nullable=True, comment="가맹점 위도")
    merchant_long = Column(Float, nullable=True, comment="가맹점 경도")

    is_fraud = Column(
        Integer,
        nullable=True,
        comment="실제 이상거래 여부. 학습/평가용 라벨"
    )

    trans_hour = Column(Integer, nullable=True, comment="거래 발생 시간대")
    trans_dayofweek = Column(Integer, nullable=True, comment="거래 발생 요일")
    trans_month = Column(Integer, nullable=True, comment="거래 발생 월")
    trans_day = Column(Integer, nullable=True, comment="거래 발생 일")

    age = Column(Integer, nullable=True, comment="거래 시점 기준 사용자 나이")
    distance_km = Column(Float, nullable=True, comment="사용자 위치와 가맹점 위치 간 거리")
    amt_zscore = Column(Float, nullable=True, comment="거래 금액의 표준화 점수")
    hour_dev = Column(Float, nullable=True, comment="사용자 평균 거래 시간 대비 편차")

    high_amt_far = Column(
        Boolean,
        nullable=True,
        comment="고액 거래이면서 장거리 거래인지 여부"
    )

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="feature 생성 시각"
    )

    updated_at = Column(
        TIMESTAMP,
        nullable=True,
        comment="feature 갱신 시각"
    )

    raw_transaction = relationship(
        "RawTransaction",
        back_populates="feature"
    )


# ============================================================
# 3. 모델 예측 결과 테이블
# ============================================================

class ModelPrediction(Base):
    """
    model_predictions

    역할:
    - ML/DL 분류 모델의 예측 결과 저장
    - 거래 1건당 예측 결과 1건
    - txn_id를 PK이자 FK로 사용
    """

    __tablename__ = "model_predictions"

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        primary_key=True,
        comment="거래 ID. raw_transactions.txn_id를 참조하는 PK/FK"
    )

    model_name = Column(
        String(100),
        nullable=True,
        comment="예측에 사용한 모델 이름"
    )

    model_version = Column(
        String(50),
        nullable=True,
        comment="예측에 사용한 모델 버전"
    )

    predicted_label = Column(
        Integer,
        nullable=True,
        comment="모델 예측 라벨. 0: 정상, 1: 사기"
    )

    fraud_probability = Column(
        Float,
        nullable=True,
        comment="모델이 예측한 사기 거래 확률"
    )

    predicted_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="모델 예측 수행 시각"
    )

    raw_transaction = relationship(
        "RawTransaction",
        back_populates="prediction"
    )


# ============================================================
# 4. 이상탐지 점수 테이블
# ============================================================

class AnomalyScore(Base):
    """
    anomaly_scores

    역할:
    - Isolation Forest, AutoEncoder 등 이상탐지 모델의 결과 저장
    - ML 분류 모델과 별도로 비정상 패턴 점수를 저장
    - 거래 1건당 이상탐지 결과 1건
    """

    __tablename__ = "anomaly_scores"

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        primary_key=True,
        comment="거래 ID. raw_transactions.txn_id를 참조하는 PK/FK"
    )

    anomaly_model_name = Column(
        String(100),
        nullable=True,
        comment="이상탐지에 사용한 모델 이름"
    )

    anomaly_score = Column(
        Float,
        nullable=True,
        comment="이상탐지 모델이 산출한 원본 이상 점수"
    )

    anomaly_score_normalized = Column(
        Float,
        nullable=True,
        comment="0~1 범위로 정규화한 이상 점수"
    )

    anomaly_label = Column(
        Integer,
        nullable=True,
        comment="이상탐지 결과 라벨. 0: 정상, 1: 이상"
    )

    scored_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="이상탐지 점수 계산 시각"
    )

    raw_transaction = relationship(
        "RawTransaction",
        back_populates="anomaly_score"
    )


# ============================================================
# 5. 최종 위험 점수 테이블
# ============================================================

class FinalScore(Base):
    """
    final_scores

    역할:
    - 모델 예측 확률과 이상탐지 점수를 조합한 최종 위험 점수 저장
    - 대시보드, alert 생성, 사용자 위험 집계의 기준값으로 사용

    기본 가정:
    - final_risk_score = ML 예측 점수 + anomaly score 조합
    """

    __tablename__ = "final_scores"

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        primary_key=True,
        comment="거래 ID. raw_transactions.txn_id를 참조하는 PK/FK"
    )

    final_risk_score = Column(
        Float,
        nullable=True,
        comment="최종 위험 점수. 모델 예측과 이상탐지 점수를 조합한 값"
    )

    risk_level = Column(
        String(20),
        nullable=True,
        comment="최종 위험 등급. 예: LOW, MEDIUM, HIGH, CRITICAL"
    )

    model_weight = Column(
        Float,
        nullable=True,
        comment="최종 점수 계산 시 ML 예측 결과에 부여한 가중치"
    )

    anomaly_weight = Column(
        Float,
        nullable=True,
        comment="최종 점수 계산 시 이상탐지 점수에 부여한 가중치"
    )

    scoring_method = Column(
        String(100),
        nullable=True,
        comment="최종 점수 계산 방식 설명. 예: weighted_sum_v1"
    )

    calculated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="최종 위험 점수 계산 시각"
    )

    raw_transaction = relationship(
        "RawTransaction",
        back_populates="final_score"
    )


# ============================================================
# 6. Alert 로그 테이블
# ============================================================

class Alert(Base):
    """
    alerts

    역할:
    - 위험 거래에 대해 발생한 경고/알림 기록 저장
    - 한 거래에서 여러 alert가 발생할 수 있으므로 txn_id는 FK만 사용
    - 관리자 검토 상태 관리 가능

    예시 alert_type:
    - HIGH_RISK_SCORE
    - HIGH_FRAUD_PROBABILITY
    - ANOMALY_DETECTED
    - HIGH_AMOUNT
    - LONG_DISTANCE
    - NIGHT_TRANSACTION
    - HIGH_AMOUNT_FAR_LOCATION
    """

    __tablename__ = "alerts"

    alert_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="alert 고유 ID"
    )

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        nullable=False,
        comment="alert가 발생한 거래 ID"
    )

    alert_type = Column(
        String(50),
        nullable=False,
        comment="alert 유형. 예: HIGH_RISK_SCORE, ANOMALY_DETECTED"
    )

    alert_message = Column(
        Text,
        nullable=True,
        comment="관리자에게 보여줄 alert 설명 메시지"
    )

    severity = Column(
        String(20),
        nullable=True,
        comment="심각도. 예: LOW, MEDIUM, HIGH, CRITICAL"
    )

    alert_status = Column(
        String(20),
        nullable=False,
        server_default=text("'OPEN'"),
        comment="처리 상태. 예: OPEN, REVIEWING, RESOLVED, IGNORED"
    )

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="alert 발생 시각"
    )

    resolved_at = Column(
        TIMESTAMP,
        nullable=True,
        comment="alert 처리 완료 시각"
    )

    raw_transaction = relationship(
        "RawTransaction",
        back_populates="alerts"
    )


# ============================================================
# 7. 사용자 위험 집계 테이블
# ============================================================

class UserRiskSummary(Base):
    """
    user_risk_summary

    역할:
    - 사용자 단위 위험도 집계 결과 저장
    - Streamlit/관리자 대시보드에서 사용자 위험도를 빠르게 조회하기 위한 테이블

    주의:
    - 현재 card_users 테이블을 생략하므로 FK 없이 user_key를 PK로 사용
    - user_key는 cc_num 원문보다는 해시값 사용 추천
    """

    __tablename__ = "user_risk_summary"

    user_key = Column(
        String(100),
        primary_key=True,
        comment="사용자 식별 키. cc_num 해시값 사용 추천"
    )

    total_txn_count = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="전체 거래 건수"
    )

    fraud_txn_count = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="실제 또는 예측 기준 사기 거래 건수"
    )

    high_risk_txn_count = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="HIGH 이상 위험 등급 거래 건수"
    )

    total_amt = Column(
        Float,
        nullable=False,
        server_default=text("0"),
        comment="전체 거래 금액 합계"
    )

    avg_amt = Column(
        Float,
        nullable=False,
        server_default=text("0"),
        comment="평균 거래 금액"
    )

    max_amt = Column(
        Float,
        nullable=False,
        server_default=text("0"),
        comment="최대 거래 금액"
    )

    avg_risk_score = Column(
        Float,
        nullable=True,
        comment="사용자 평균 최종 위험 점수"
    )

    max_risk_score = Column(
        Float,
        nullable=True,
        comment="사용자 최대 최종 위험 점수"
    )

    avg_distance_km = Column(
        Float,
        nullable=True,
        comment="사용자 평균 거래 거리"
    )

    high_amt_far_count = Column(
        Integer,
        nullable=False,
        server_default=text("0"),
        comment="고액+장거리 거래 발생 횟수"
    )

    latest_risk_level = Column(
        String(20),
        nullable=True,
        comment="최근 기준 사용자 위험 등급"
    )

    last_txn_at = Column(
        DateTime,
        nullable=True,
        comment="사용자의 마지막 거래 시각"
    )

    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="사용자 위험 집계 갱신 시각"
    )


# ============================================================
# 8. 대시보드 출력용 거래 테이블
# ============================================================

class DashboardTransaction(Base):
    """
    dashboard_transactions

    역할:
    - Streamlit 또는 프론트엔드에서 바로 출력하기 위한 화면용 테이블
    - 여러 테이블을 매번 조인하지 않도록 필요한 값만 복사해 저장
    - 실제 SQL VIEW가 아니라 조회 최적화용 테이블로 운영

    주의:
    - 원천 데이터는 raw_transactions, transaction_features, final_scores에 있음
    - 이 테이블은 화면 표시 편의를 위한 파생/복사 테이블
    """

    __tablename__ = "dashboard_transactions"

    dashboard_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="대시보드 출력용 row ID"
    )

    txn_id = Column(
        BigInteger,
        ForeignKey("raw_transactions.txn_id", ondelete="CASCADE"),
        nullable=False,
        comment="화면에 표시할 거래 ID"
    )

    trans_date_trans_time = Column(
        DateTime,
        nullable=True,
        comment="거래 발생 일시"
    )

    category = Column(String(100), nullable=True, comment="거래 카테고리")
    amt = Column(Float, nullable=True, comment="거래 금액")
    gender = Column(String(1), nullable=True, comment="사용자 성별")
    state = Column(String(20), nullable=True, comment="사용자 지역")

    age = Column(Integer, nullable=True, comment="사용자 나이")
    distance_km = Column(Float, nullable=True, comment="사용자-가맹점 거리")
    high_amt_far = Column(Boolean, nullable=True, comment="고액+장거리 거래 여부")

    actual_label = Column(Integer, nullable=True, comment="실제 라벨")
    predicted_label = Column(Integer, nullable=True, comment="모델 예측 라벨")
    fraud_probability = Column(Float, nullable=True, comment="모델 사기 예측 확률")

    anomaly_score_normalized = Column(Float, nullable=True, comment="정규화 이상탐지 점수")
    final_risk_score = Column(Float, nullable=True, comment="최종 위험 점수")
    risk_level = Column(String(20), nullable=True, comment="최종 위험 등급")

    alert_status = Column(
        String(20),
        nullable=True,
        comment="alert 처리 상태 요약"
    )

    display_status = Column(
        String(30),
        nullable=False,
        server_default=text("'VISIBLE'"),
        comment="화면 표시 상태. 예: VISIBLE, HIDDEN"
    )

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="대시보드 테이블 생성 시각"
    )