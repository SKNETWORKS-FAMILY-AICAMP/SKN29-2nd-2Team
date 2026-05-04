# 📦 산출물 3: 모델 메타데이터

## 모델 메타데이터

### 기본 정보

| 항목              | 내용                       |
| --------------- | ------------------------ |
| 모델명             | XGBoost Tuned Classifier |
| 버전              | v1.0.0                   |
| 저장일             | 2026-05-03               |
| 작성자             | FinGuard 프로젝트 팀          |
| 파일명             | `xgboost_tuned.pkl`      |
| 모델 유형           | 지도학습 기반 이진 분류 모델         |
| 예측 대상           | 신용카드 거래의 이상거래 여부         |
| Target          | `is_fraud`               |
| Threshold       | 0.58                     |
| Threshold 선정 기준 | F2-Score 최대화             |

---

### 학습 환경

| 항목           | 버전    |
| ------------ | ----- |
| Python       | 3.13.12|
| pandas       | 2.4.4 |
| numpy        | 2.4.4 |
| scikit-learn | 1.7.1 |
| xgboost      | 3.2.0 |
| lightgbm     | 4.6.0 |
| catboost     | 1.2.8 |
| torch        | 2.9.1 |
| joblib       | 1.5.3 |

---

### 모델 성능 (최종)

| 지표        |      값 |
| --------- | -----: |
| Precision | 0.9166 |
| Recall    | 0.9367 |
| F1-Score  | 0.9265 |
| F2-Score  | 0.9326 |
| PR-AUC    | 0.9717 |
| ROC-AUC   | 0.9994 |

### 성능 해석

* Recall이 0.9367로 높게 나타나 실제 이상거래를 탐지하는 성능이 우수하다.
* Precision이 0.9166으로 나타나 정상 거래를 이상거래로 잘못 판단하는 오탐도 일정 수준 관리 가능하다.
* PR-AUC가 0.9717로 높아, 이상거래 비율이 낮은 불균형 데이터 환경에서도 안정적인 탐지 성능을 보였다.
* F2-Score 기준 최적 Threshold는 0.58로 설정하였다.

---

### 하이퍼파라미터

최종 모델은 Optuna를 활용하여 XGBoost 모델의 하이퍼파라미터를 탐색하였다.

| 파라미터             | 값     |
| ---------------- | ----- |
| n_estimators     | 464 |
| max_depth        | 9 |
| learning_rate    | 0.21 |
| subsample        | 0.81 |
| colsample_bytree | 0.80 |
| min_child_weight | 3 |
| gamma            | 0.73 |
| reg_alpha        | 0.32 |
| reg_lambda       | 5.96 |
| scale_pos_weight | 적용    |
| random_state     | 42    |

---

### 입력 데이터 스펙

#### 입력 특성 수

| 구분     | 내용            |
| ------ | ------------- |
| 수치형 특성 | 17개           |
| 범주형 특성 | 4개            |
| Target | `is_fraud`    |
| 입력 데이터 | 전처리 완료 거래 데이터 |

---

#### 수치형 입력 특성

| 특성명               | 설명                |
| ----------------- | ----------------- |
| `amt`             | 거래 금액             |
| `zip`             | 우편번호              |
| `lat`             | 고객 위도             |
| `long`            | 고객 경도             |
| `city_pop`        | 고객 거주 도시 인구       |
| `unix_time`       | 거래 Unix timestamp |
| `merch_lat`       | 가맹점 위도            |
| `merch_long`      | 가맹점 경도            |
| `trans_hour`      | 거래 시간             |
| `trans_dayofweek` | 거래 요일             |
| `trans_month`     | 거래 월              |
| `trans_day`       | 거래 일              |
| `age`             | 고객 나이             |
| `distance_km`     | 고객-가맹점 간 거리       |
| `amt_zscore`      | 고객 평균 대비 거래 금액 편차 |
| `hour_dev`        | 평균 거래 시간 대비 편차    |
| `high_amt_far`    | 고액 + 장거리 거래 여부    |

---

#### 범주형 입력 특성

| 특성명        | 설명      |
| ---------- | ------- |
| `merchant` | 가맹점명    |
| `category` | 거래 업종   |
| `gender`   | 고객 성별   |
| `state`    | 고객 거주 주 |

---

### 필수 전처리

| 처리 항목       | 설명                                                             |
| ----------- | -------------------------------------------------------------- |
| 날짜 파생 변수 생성 | `trans_hour`, `trans_dayofweek`, `trans_month`, `trans_day` 생성 |
| 나이 계산       | `dob` 기반 `age` 생성                                              |
| 거리 계산       | 고객 좌표와 가맹점 좌표 기반 `distance_km` 생성                              |
| 금액 편차 계산    | 고객별 평균/표준편차 기반 `amt_zscore` 생성                                 |
| 시간 편차 계산    | 거래 시간 기반 `hour_dev` 생성                                         |
| 복합 위험 변수 생성 | `high_amt_far` 생성                                              |
| 범주형 변수 처리   | 모델 파이프라인에 맞는 인코딩 적용                                            |
| 입력 컬럼 순서 유지 | 학습 시 사용한 Feature 순서와 동일하게 입력                                   |

---

### 예측값 해석

| 출력값                   | 의미        |
| --------------------- | --------- |
| `0`                   | 정상 거래     |
| `1`                   | 이상거래      |
| `predict_proba[:, 1]` | 이상거래일 확률  |
| Threshold 미만          | 정상 거래로 판단 |
| Threshold 이상          | 이상거래로 판단  |

### 최종 Threshold

| 항목           |                          값 |
| ------------ | -------------------------: |
| 기본 Threshold |                       0.50 |
| 최종 Threshold |                       0.58 |
| 선정 기준        |               F2-Score 최대화 |
| 목적           | Fraud 미탐 최소화와 오탐 관리의 균형 확보 |

---

### Final Score 연계

본 프로젝트에서는 XGBoost 모델의 예측 확률을 단독으로 사용하지 않고, Rule 기반 Risk Score 및 AutoEncoder 기반 Anomaly Score와 결합하여 최종 위험도를 산출하였다.

| 점수            | 설명                   | 가중치 |
| ------------- | -------------------- | --: |
| Risk Score    | Rule 기반 위험 점수        | 40% |
| ML Score      | XGBoost 예측 점수        | 40% |
| Anomaly Score | AutoEncoder 기반 이상 점수 | 20% |

```text
Final Score = Risk Score × 0.4 + ML Score × 0.4 + Anomaly Score × 0.2
```

| Final Score 활용 | 설명                      |
| -------------- | ----------------------- |
| Pass           | 위험도가 낮은 거래              |
| Review         | 검토가 필요한 거래              |
| Block          | 차단 또는 우선 대응이 필요한 고위험 거래 |

---

### 모델 로드 및 예측 예시

```python
import joblib
import pandas as pd

# 모델 로드
model = joblib.load("model/xgboost_tuned.pkl")

# 신규 거래 데이터 예시
new_transaction = pd.DataFrame([{
    "amt": 250.0,
    "zip": 28611,
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 3495,
    "unix_time": 1325376018,
    "merch_lat": 36.011293,
    "merch_long": -82.048315,
    "trans_hour": 23,
    "trans_dayofweek": 2,
    "trans_month": 1,
    "trans_day": 1,
    "age": 35,
    "distance_km": 85.3,
    "amt_zscore": 2.7,
    "hour_dev": 1.8,
    "high_amt_far": 1,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "gender": "F",
    "state": "NC"
}])

# 이상거래 확률 예측
fraud_prob = model.predict_proba(new_transaction)[:, 1][0]

# 최종 threshold 적용
threshold = 0.58
prediction = 1 if fraud_prob >= threshold else 0

print("이상거래 확률:", round(fraud_prob, 4))
print("예측 결과:", "이상거래" if prediction == 1 else "정상거래")
```

---

### 재현 방법

1. 원본 데이터 준비

   * `data/cc_fraud_train.csv`
   * `data/cc_fraud_test.csv`

2. 데이터 전처리 실행

   * `notebook/preprocessing.ipynb`

3. 머신러닝 모델 학습 및 비교

   * `notebook/model_comparison.ipynb`
   * `notebook/train_ml_score.ipynb`

4. XGBoost Threshold 탐색

   * `notebook/xgboost_threshold_search.ipynb`

5. 이상 탐지 모델 학습

   * `notebook/train_anomaly_model.ipynb`

6. Final Score 통합

   * `notebook/total_score.ipynb`

7. 최종 모델 및 점수 데이터 확인

   * `model/xgboost_tuned.pkl`
   * `data/total_scored_data.csv`

---

### 저장 파일 목록

| 파일명                                      | 설명                                   |
| ---------------------------------------- | ------------------------------------ |
| `model/xgboost_tuned.pkl`                | 최종 XGBoost 튜닝 모델                     |
| `model/xgboost_model.pkl`                | 기본 XGBoost 모델                        |
| `model/randomforest_model.pkl`           | Random Forest 모델                     |
| `model/lightgbm_model.pkl`               | LightGBM 모델                          |
| `model/catboost_model.pkl`               | CatBoost 모델                          |
| `notebook/model/autoencoder_anomaly.pth` | AutoEncoder 이상 탐지 모델                 |
| `notebook/model/scaler_anomaly.pkl`      | AutoEncoder 입력용 스케일러                 |
| `data/total_scored_data.csv`             | Risk, ML, Anomaly, Final Score 통합 결과 |
| `output/threshold_results.csv`           | Threshold별 성능 결과                     |
| `output/feature_importance_XGBoost.png`  | XGBoost Feature Importance 이미지       |

---

### 알려진 한계점

| 항목       | 내용                                                          |
| -------- | ----------------------------------------------------------- |
| 데이터 한계   | 공개 데이터 기반이므로 실제 금융사 운영 데이터와 차이가 있을 수 있음                     |
| 불균형 데이터  | Fraud 비율이 낮아 Threshold 설정에 따라 오탐/미탐 trade-off가 발생           |
| 실시간 처리   | 현재는 Streamlit 기반 분석 및 시뮬레이션 중심이며 실시간 API 연동은 추가 필요          |
| 모델 해석    | Feature Importance는 확인했으나 SHAP 기반 상세 해석은 추가 가능              |
| 데이터 드리프트 | 거래 패턴 변화 시 주기적인 재학습 필요                                      |
| 운영 정책    | Final Score 기준 Pass / Review / Block 구간은 실제 운영 정책에 맞게 조정 필요 |

---

### 개선 방향

| 개선 항목               | 설명                                     |
| ------------------- | -------------------------------------- |
| 실시간 API 연동          | FastAPI 등을 활용해 실시간 거래 예측 API 구성        |
| 모델 재학습 자동화          | 신규 거래 데이터 누적에 따른 주기적 재학습 파이프라인 구축      |
| SHAP 분석 추가          | 개별 거래별 이상 판단 근거 설명 강화                  |
| 비용 기반 Threshold 최적화 | 오탐 비용과 미탐 비용을 반영한 운영 Threshold 설정      |
| 알림 시스템 연계           | Block 또는 Review 거래 발생 시 관리자 알림 기능 추가   |
| DB 기반 운영 강화         | 모델 예측 결과와 Alert 이력을 DB에 저장하여 운영 추적성 확보 |
