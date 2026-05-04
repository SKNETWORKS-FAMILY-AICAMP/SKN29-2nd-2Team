# scripts/download_kaggle_data.py

# kaggle API로 데이터 받아올 경우,
# 1. pip install kaggle
# 2. kaggle API 키 발급 및 환경변수 등록 필요.
# 3. python -m scripts.download_kaggle_data로 실행.

import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


load_dotenv()


DATA_DIR = Path("data")
DATASET_NAME = "kaushalnandania/credit-card-fraud-detection"

# 우리가 프로젝트에서 사용할 표준 파일명
TRAIN_TARGET_NAME = "cc_fraud_train.csv"
TEST_TARGET_NAME = "cc_fraud_test.csv"


def list_data_files():
    """
    data 폴더 안의 파일 목록 확인
    """

    print("[INFO] Current files in data folder:")

    for file in DATA_DIR.iterdir():
        print(f" - {file.name}")


def rename_downloaded_files():
    """
    Kaggle에서 받은 원본 파일명을 프로젝트 표준 파일명으로 변경

    프로젝트 진행 기간 기준 원본 파일명:
    - train.csv
    - test.csv
    - fraudTrain.csv
    - fraudTest.csv
    """

    rename_rules = {
        "fraudTrain.csv": TRAIN_TARGET_NAME,
        "fraudTest.csv": TEST_TARGET_NAME,
        "train.csv": TRAIN_TARGET_NAME,
        "test.csv": TEST_TARGET_NAME,
    }

    for original_name, target_name in rename_rules.items():
        original_path = DATA_DIR / original_name
        target_path = DATA_DIR / target_name

        if original_path.exists():
            # 이미 target 파일이 있으면 삭제 후 교체
            if target_path.exists():
                target_path.unlink()

            original_path.rename(target_path)

            print(f"[INFO] Renamed: {original_name} -> {target_name}")


def download_dataset():
    """
    Kaggle 데이터셋 다운로드 → 압축 해제 → 파일명 확인 → 자동 rename
    """

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

    DATA_DIR.mkdir(exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("[INFO] Downloading Kaggle dataset...")

    api.dataset_download_files(
        DATASET_NAME,
        path=str(DATA_DIR),
        unzip=False
    )

    print("[INFO] Download complete.")

    zip_files = list(DATA_DIR.glob("*.zip"))

    if not zip_files:
        print("[WARN] No zip file found.")
    else:
        for zip_path in zip_files:
            print(f"[INFO] Extracting: {zip_path.name}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)

            zip_path.unlink()
            print(f"[INFO] Removed zip file: {zip_path.name}")

    # 2. 다운로드 파일명 확인
    list_data_files()

    # 3. 자동 rename
    rename_downloaded_files()

    print("[INFO] Final data files:")
    list_data_files()


if __name__ == "__main__":
    download_dataset()