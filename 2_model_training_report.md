# 📄 산출물 2: 인공지능 학습 결과서

## 1. 모델링 전략

### 1-1. 평가 지표 선정

본 프로젝트는 신용카드 이상거래 탐지 문제를 다룬다.
이상거래 데이터는 정상 거래에 비해 Fraud 비율이 매우 낮은 불균형 데이터이므로, Accuracy만으로 모델을 평가하기 어렵다.

| 지표        | 선정 이유                                |
| --------- | ------------------------------------ |
| Recall    | 실제 이상거래를 놓치지 않는 것이 중요하기 때문에 최우선으로 고려 |
| Precision | 정상 거래를 이상거래로 잘못 판단하는 오탐을 관리하기 위해 필요  |
| F1-Score  | Precision과 Recall의 균형 확인             |
| F2-Score  | Recall에 더 높은 가중치를 두어 Fraud 미탐 최소화    |
| ROC-AUC   | 전체적인 분류 구분력 확인                       |
| PR-AUC    | 불균형 데이터에서 Fraud 탐지 성능을 평가하는 데 적합     |
| Accuracy  | 참고 지표로만 활용                           |

> 신용카드 이상거래 탐지에서는 False Negative, 즉 실제 Fraud 거래를 정상 거래로 판단하는 경우가 큰 손실로 이어질 수 있다.
> 따라서 Recall과 F2-Score를 중요하게 보고, Precision과 PR-AUC를 함께 확인하였다.

---

### 1-2. 후보 모델 선정

| 모델                  | 유형    | 선정 근거                                      |
| ------------------- | ----- | ------------------------------------------ |
| Logistic Regression | 선형 모델 | 기준선(Baseline) 모델로 활용, 해석이 쉬움               |
| Decision Tree       | 트리 모델 | 단순한 규칙 기반 분류 구조 확인 가능                      |
| Random Forest       | 앙상블   | 불균형 데이터에 비교적 강건하고 Feature Importance 확인 가능 |
| Extra Trees         | 앙상블   | 무작위성을 강화한 트리 기반 모델                         |
| Gradient Boosting   | 부스팅   | 순차적으로 오차를 보정하는 모델                          |
| XGBoost             | 부스팅   | 불균형 분류 문제에서 높은 성능을 기대할 수 있음                |
| LightGBM            | 부스팅   | 빠른 학습 속도와 높은 성능                            |
| CatBoost            | 부스팅   | 범주형 변수 처리에 강점                              |
| Soft Voting         | 앙상블   | 여러 모델의 예측 확률을 결합                           |
| Hard Voting         | 앙상블   | 여러 모델의 예측 결과를 다수결로 결합                      |
| Stacking            | 앙상블   | 여러 모델의 예측 결과를 메타 모델로 재학습                   |

---

### 1-3. 실험 계획

* 기본 전처리 데이터를 기반으로 여러 머신러닝 모델을 학습하였다.
* Validation Set 기준으로 F1, ROC-AUC, PR-AUC를 비교하였다.
* 불균형 데이터 특성을 고려하여 PR-AUC와 Recall을 중요하게 확인하였다.
* 최종 후보 모델로 XGBoost를 선정하고 Optuna 기반 하이퍼파라미터 튜닝을 수행하였다.
* Threshold를 조정하여 F2-Score가 최대가 되는 분류 기준을 탐색하였다.
* 최종적으로 XGBoost 예측 점수와 Rule 기반 Risk Score, AutoEncoder 기반 Anomaly Score를 결합하여 Final Score를 구성하였다.

---

## 2. 머신러닝 모델 학습 결과

### 2-1. Logistic Regression

**하이퍼파라미터**

| 파라미터     | 값         |
| ------------ | --------- |
| max_iter     | 1000      |
| solver       | liblinear |
| class_weight | balanced  |
| random_state | 42        |

**성능 결과**

| 지표       | Validation |
| -------- | ---------: |
| F1-Score |     0.0765 |
| ROC-AUC  |     0.9374 |
| PR-AUC   |     0.2329 |

**특이사항**

* 기준선 모델로 사용하였다.
* 선형 모델이기 때문에 복잡한 비선형 Fraud 패턴을 충분히 반영하는 데 한계가 있었다.

---

### 2-2. Decision Tree

**하이퍼파라미터**

| 파라미터         | 값        |
| ------------ | -------- |
| max_depth    | 10       |
| class_weight | balanced |
| random_state | 42       |

**성능 결과**

| 지표       | Validation |
| -------- | ---------: |
| F1-Score |     0.2809 |
| ROC-AUC  |     0.9795 |
| PR-AUC   |     0.5203 |

**특이사항**

* 단일 트리 구조로 해석은 쉬우나, 일반화 성능과 안정성 측면에서 한계가 있었다.

---

### 2-3. Random Forest

**하이퍼파라미터**

| 파라미터         | 값        |
| ------------ | -------- |
| n_estimators | 300      |
| max_depth    | 15       |
| class_weight | balanced |
| n_jobs       | -1       |
| random_state | 42       |

**성능 결과**

| 지표       | Validation |
| -------- | ---------: |
| F1-Score |     0.3150 |
| ROC-AUC  |     0.9794 |
| PR-AUC   |     0.6838 |

**특이사항**

* 트리 앙상블 기반으로 안정적인 성능을 보였으나, 최종 모델 후보인 XGBoost 대비 PR-AUC가 낮았다.

---

### 2-4. 주요 부스팅 모델 비교

| 모델                | F1-Score | ROC-AUC | PR-AUC |
| ----------------- | -------: | ------: | -----: |
| Gradient Boosting |   0.8677 |  0.9946 | 0.8777 |
| XGBoost           |   0.5027 |  0.9987 | 0.9284 |
| LightGBM          |   0.4576 |  0.9976 | 0.9188 |
| CatBoost          |   0.3876 |  0.9984 | 0.8844 |

**해석**

* Gradient Boosting은 F1-Score가 높게 나타났다.
* XGBoost는 PR-AUC와 ROC-AUC가 높아 불균형 데이터에서 Fraud 탐지 후보 모델로 적합하다고 판단하였다.
* 최종 모델은 PR-AUC와 Recall 중심 운영 가능성을 고려하여 XGBoost를 중심으로 튜닝하였다.

---

### 2-5. 앙상블 모델 비교

| 모델          | F1-Score | ROC-AUC | PR-AUC |
| ----------- | -------: | ------: | -----: |
| Stacking    |   0.4411 |  0.9986 | 0.9182 |
| Soft Voting |   0.4990 |  0.9976 | 0.8808 |
| Hard Voting |   0.5037 |       - |      - |

**해석**

* Stacking과 Voting 모델도 높은 성능을 보였으나, XGBoost 단일 모델 대비 PR-AUC 개선 폭이 크지 않았다.
* 모델 운영 및 해석, 튜닝 편의성을 고려하여 최종 모델은 XGBoost로 선정하였다.

---

### 2-6. XGBoost 하이퍼파라미터 튜닝

**튜닝 방법**

| 항목           | 내용                  |
| ------------ | ------------------- |
| 튜닝 도구        | Optuna              |
| Trial 수      | 100                 |
| 최적화 기준       | PR-AUC              |
| 불균형 보정       | scale_pos_weight 적용 |
| random_state | 42                  |

**탐색 파라미터**

| 파라미터             | 탐색 범위       |
| ---------------- | ----------- |
| n_estimators     | 100 ~ 500   |
| max_depth        | 3 ~ 10      |
| learning_rate    | 0.01 ~ 0.3  |
| subsample        | 0.6 ~ 1.0   |
| colsample_bytree | 0.6 ~ 1.0   |
| min_child_weight | 1 ~ 10      |
| gamma            | 0.0 ~ 1.0   |
| reg_alpha        | 1e-4 ~ 10.0 |
| reg_lambda       | 1e-4 ~ 10.0 |

**튜닝 전/후 비교**

| 지표      |   튜닝 전 |   튜닝 후 |      개선 |
| ------- | -----: | -----: | ------: |
| PR-AUC  | 0.9284 | 0.9717 | +0.0433 |
| ROC-AUC | 0.9987 | 0.9994 | +0.0007 |

**튜닝 후 주요 성능**

| 지표        |      값 |
| --------- | -----: |
| Precision | 0.9166 |
| Recall    | 0.9367 |
| F1-Score  | 0.9265 |
| F2-Score  | 0.9326 |
| PR-AUC    | 0.9717 |
| ROC-AUC   | 0.9994 |

---

## 3. 딥러닝 모델 학습 결과

### 3-1. AutoEncoder 모델 개요

본 프로젝트에서는 지도학습 기반 분류 모델 외에, 정상 거래 패턴과 다른 비정상 거래를 탐지하기 위해 AutoEncoder 기반 이상 탐지 모델을 추가로 활용하였다.

AutoEncoder는 입력 데이터를 압축한 뒤 다시 복원하는 구조이며, 복원 오차가 클수록 정상 패턴에서 벗어난 거래로 판단할 수 있다.

---

### 3-2. 모델 구조

**네트워크 아키텍처**

```text
입력층 (17개 수치형 특성)
    │
    ├─ Linear(17 → 16)
    ├─ Linear(16 → 8)
    ├─ Linear(8 → 16)
    └─ Linear(16 → 17)
```

| 항목      | 내용                                       |
| ------- | ---------------------------------------- |
| 모델 유형   | AutoEncoder                              |
| 입력 특성 수 | 17개 수치형 Feature                          |
| 목적      | Reconstruction Error 기반 이상 패턴 탐지         |
| 저장 모델   | `notebook/model/autoencoder_anomaly.pth` |
| 스케일러    | `notebook/model/scaler_anomaly.pkl`      |

---

### 3-3. 학습 설정

| 파라미터   | 값                       |
| ------ | ----------------------- |
| 손실 함수  | MSELoss                 |
| 옵티마이저  | Adam                    |
| 입력 데이터 | 수치형 Feature             |
| 스케일링   | StandardScaler 적용       |
| 이상 점수  | Reconstruction Error 기반 |
| 최종 활용  | Anomaly Score 산출        |

---

### 3-4. AutoEncoder 활용 결과

| 결과             | 설명                          |
| -------------- | --------------------------- |
| Anomaly Score  | 복원 오차 기반 이상 점수              |
| 활용 목적          | XGBoost가 포착하지 못하는 비정상 패턴 보완 |
| Final Score 반영 | Anomaly Score를 20% 가중치로 반영  |

AutoEncoder는 최종 분류 모델이라기보다, Final Score를 구성하는 보조 이상 탐지 모듈로 활용하였다.

---

## 4. 모델 종합 비교 및 최적 모델 선정

### 4-1. 전체 모델 성능 비교표

| 모델                  | F1-Score | ROC-AUC | PR-AUC |
| ------------------- | -------: | ------: | -----: |
| Logistic Regression |   0.0765 |  0.9374 | 0.2329 |
| Decision Tree       |   0.2809 |  0.9795 | 0.5203 |
| Random Forest       |   0.3150 |  0.9794 | 0.6838 |
| Extra Trees         |   0.0716 |  0.9170 | 0.4128 |
| Gradient Boosting   |   0.8677 |  0.9946 | 0.8777 |
| XGBoost             |   0.5027 |  0.9987 | 0.9284 |
| LightGBM            |   0.4576 |  0.9976 | 0.9188 |
| CatBoost            |   0.3876 |  0.9984 | 0.8844 |
| Stacking            |   0.4411 |  0.9986 | 0.9182 |
| Soft Voting         |   0.4990 |  0.9976 | 0.8808 |
| Hard Voting         |   0.5037 |       - |      - |
| XGBoost Tuned       |   0.9265 |  0.9994 | 0.9717 |

---

### 4-2. 지표별 최고 성능 모델

| 지표       | 최고 모델         |      값 |
| -------- | ------------- | -----: |
| F1-Score | XGBoost Tuned | 0.9265 |
| F2-Score | XGBoost Tuned | 0.9326 |
| PR-AUC   | XGBoost Tuned | 0.9717 |
| ROC-AUC  | XGBoost Tuned | 0.9994 |

---

### 4-3. ROC Curve 비교

<p align="center">
  <img src="output/roc_curves.png" width="90%" alt="ROC Curves">
</p>

---

### 4-4. Precision-Recall Curve 비교

<p align="center">
  <img src="output/pr_curves.png" width="90%" alt="Precision Recall Curves">
</p>

---

### 4-5. 최적 모델 선정

**선정 모델**: XGBoost Tuned

**선정 근거**

1. PR-AUC 0.9717로 불균형 데이터 환경에서 우수한 탐지 성능을 보였다.
2. Recall 0.9367로 실제 Fraud 거래를 높은 비율로 탐지하였다.
3. F2-Score 0.9326으로 미탐 최소화 관점에서 높은 성능을 보였다.
4. ROC-AUC 0.9994로 정상 거래와 Fraud 거래를 구분하는 능력이 우수했다.
5. 모델 저장 및 서비스 연동이 용이하여 Streamlit 대시보드와 시뮬레이션 페이지에 적용하기 적합했다.

---

### 4-6. 최적 임계값 결정

XGBoost 모델의 기본 threshold는 0.5이지만, 이상거래 탐지에서는 Recall과 Precision 간의 trade-off를 고려해야 한다.

본 프로젝트에서는 Fraud 미탐을 줄이는 것이 중요하다고 판단하여 F2-Score 기준으로 최적 threshold를 탐색하였다.

| 항목           |        값 |
| ------------ | -------: |
| 기본 Threshold |     0.50 |
| 최적 Threshold |     0.58 |
| 기준 지표        | F2-Score |
| 최적 F2-Score  |   0.9326 |

<p align="center">
  <img src="output/threshold_f2.png" width="90%" alt="Threshold F2">
</p>

---

### 4-7. Threshold Trade-off 분석

<p align="center">
  <img src="output/threshold_comparison.png" width="90%" alt="Threshold Comparison">
</p>

| Threshold 변화 | 영향                         |
| ------------ | -------------------------- |
| 낮게 설정        | Recall 증가, 미탐 감소, 오탐 증가    |
| 높게 설정        | Precision 증가, 오탐 감소, 미탐 증가 |

따라서 서비스 운영 환경에서는 보수적 탐지, 균형 탐지, 공격적 탐지 전략에 따라 threshold를 조정할 수 있다.

---

## 5. 모델 해석

### 5-1. 특성 중요도

최종 모델인 XGBoost 기준 Feature Importance를 확인하였다.

<p align="center">
  <img src="output/feature_importance_XGBoost.png" width="90%" alt="Feature Importance XGBoost">
</p>

주요 Feature는 다음과 같다.

| 주요 Feature       | 해석                             |
| ---------------- | ------------------------------ |
| `amt`            | 거래 금액은 Fraud 탐지에 중요한 변수        |
| `category` 관련 변수 | 특정 업종에서 Fraud 패턴이 다르게 나타날 수 있음 |
| `trans_hour`     | 시간대별 이상거래 패턴 반영                |
| `amt_zscore`     | 고객 평균 대비 고액 거래 여부 반영           |
| `hour_dev`       | 평소 거래 시간과 다른 거래 패턴 반영          |
| `distance_km`    | 고객 위치와 가맹점 위치 간 거리 반영          |
| `high_amt_far`   | 고액 + 장거리 복합 위험 반영              |

---

### 5-2. Final Score 기반 해석

본 프로젝트는 XGBoost 단일 모델 결과만 사용하지 않고, 다음 세 가지 점수를 결합하여 Final Score를 구성하였다.

| 점수            | 설명                   | 가중치 |
| ------------- | -------------------- | --: |
| Risk Score    | Rule 기반 위험 점수        | 40% |
| ML Score      | XGBoost 기반 예측 점수     | 40% |
| Anomaly Score | AutoEncoder 기반 이상 점수 | 20% |

```text
Final Score = Risk Score × 0.4 + ML Score × 0.4 + Anomaly Score × 0.2
```

샘플 데이터 기준 정상 거래와 이상거래의 평균 점수 차이는 다음과 같다.

| 구분            | 정상 거래 평균 | 이상거래 평균 |
| ------------- | -------: | ------: |
| Risk Score    |    27.35 |   65.47 |
| ML Score      |     0.05 |   98.23 |
| Anomaly Score |    32.05 |   58.70 |
| Final Score   |    17.37 |   77.22 |

이 결과는 이상거래가 정상 거래보다 Rule 기반 위험도, ML 예측 점수, 이상 탐지 점수에서 모두 높게 나타나는 경향이 있음을 보여준다.

---

### 5-3. 고위험 거래 프로파일

Final Score와 주요 Feature를 기준으로 고위험 거래는 다음과 같은 특징을 가진다.

* 거래 금액이 평소보다 큰 경우
* 심야 시간대에 발생한 거래
* 고객 위치와 가맹점 위치 간 거리가 먼 거래
* 고액 거래와 장거리 거래가 동시에 발생한 경우
* 고객의 평균 거래 시간 또는 금액 패턴에서 벗어난 거래
* XGBoost 예측 확률이 높은 거래
* AutoEncoder 복원 오차가 큰 거래

---

## 6. 결론

본 프로젝트에서는 신용카드 이상거래 탐지를 위해 다양한 머신러닝 모델을 비교하고, 최종적으로 XGBoost 모델을 선정하였다.

XGBoost 모델은 PR-AUC 0.9717, ROC-AUC 0.9994, Recall 0.9367, F2-Score 0.9326으로 불균형 데이터 환경에서도 우수한 탐지 성능을 보였다.

또한 Rule 기반 Risk Score와 AutoEncoder 기반 Anomaly Score를 함께 활용하여 단일 모델의 한계를 보완하였다.

최종적으로 Risk Score, ML Score, Anomaly Score를 결합한 Final Score를 통해 고위험 거래를 보다 안정적으로 선별할 수 있는 구조를 마련하였다.

향후에는 실제 운영 데이터 기반 재학습, SHAP 기반 모델 해석, 실시간 API 연동, 운영 비용 기반 threshold 최적화를 통해 이상거래 탐지 시스템의 실용성을 더욱 높일 수 있다.
