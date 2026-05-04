# db/load_raw_data.py

import math
import pandas as pd
from tqdm import tqdm

from db.database import SessionLocal
from db.models import RawTransaction


def preprocess_raw_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    원본 CSV chunk 단위 전처리 함수

    역할:
    - CSV에 포함된 불필요 index 컬럼 제거
    - 원본 컬럼명을 DB 컬럼명에 맞게 변경
    - 날짜 타입 변환
    - zip_code 문자열 변환
    - NaN 값을 None으로 변환
    """

    if "Unnamed: 0" in chunk.columns:
        chunk = chunk.drop(columns=["Unnamed: 0"])

    chunk = chunk.rename(columns={
        "first": "first_name",
        "last": "last_name",
        "zip": "zip_code",
        "lat": "customer_lat",
        "long": "customer_long",
        "merch_lat": "merchant_lat",
        "merch_long": "merchant_long",
    })

    chunk["trans_date_trans_time"] = pd.to_datetime(
        chunk["trans_date_trans_time"],
        errors="coerce"
    )

    chunk["dob"] = pd.to_datetime(
        chunk["dob"],
        errors="coerce"
    ).dt.date

    chunk["zip_code"] = chunk["zip_code"].astype(str)
    chunk["cc_num"] = chunk["cc_num"].astype(str)

    chunk = chunk.where(pd.notnull(chunk), None)

    return chunk


def chunk_to_objects(
    chunk: pd.DataFrame,
    data_split: str,
    data_source: str
) -> list[RawTransaction]:
    """
    DataFrame chunk를 RawTransaction ORM 객체 리스트로 변환

    역할:
    - raw_transactions 테이블에 insert할 객체 생성
    - raw_transactions는 원본 보관용이므로 파생 컬럼은 넣지 않음
    """

    objects = []

    for _, row in chunk.iterrows():
        obj = RawTransaction(
            data_split=data_split,
            data_source=data_source,

            trans_date_trans_time=row["trans_date_trans_time"],
            cc_num=row["cc_num"],

            merchant=row["merchant"],
            category=row["category"],
            amt=row["amt"],

            first_name=row["first_name"],
            last_name=row["last_name"],
            gender=row["gender"],

            street=row["street"],
            city=row["city"],
            state=row["state"],
            zip_code=row["zip_code"],

            customer_lat=row["customer_lat"],
            customer_long=row["customer_long"],
            city_pop=row["city_pop"],

            job=row["job"],
            dob=row["dob"],

            trans_num=row["trans_num"],
            unix_time=row["unix_time"],

            merchant_lat=row["merchant_lat"],
            merchant_long=row["merchant_long"],

            is_fraud=row["is_fraud"],
        )

        objects.append(obj)

    return objects


def load_raw_data(
    csv_path: str,
    data_split: str,
    data_source: str = "kaggle",
    chunk_size: int = 10000
):
    """
    원본 CSV 데이터를 raw_transactions 테이블에 적재하는 함수

    처리 흐름:
    1. CSV 전체 row 수 확인
    2. chunk 단위로 CSV 읽기
    3. 컬럼명/타입 정리
    4. ORM 객체 변환
    5. bulk insert
    """

    print("[INFO] Loading raw data...")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] Data split: {data_split}")

    total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
    total_chunks = math.ceil(total_rows / chunk_size)

    print(f"[INFO] Total rows: {total_rows}")
    print(f"[INFO] Total chunks: {total_chunks}")

    db = SessionLocal()
    total_inserted = 0

    try:
        chunk_reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            index_col=0
        )

        progress_bar = tqdm(
            chunk_reader,
            total=total_chunks,
            desc=f"Loading {data_split} raw data"
        )

        for chunk in progress_bar:
            chunk = preprocess_raw_chunk(chunk)
            objects = chunk_to_objects(chunk, data_split, data_source)

            db.bulk_save_objects(objects)
            db.commit()

            total_inserted += len(objects)

            progress_bar.set_postfix({
                "Inserted": total_inserted
            })

        print("[INFO] Raw data insert complete.")
        print(f"[INFO] Total inserted rows: {total_inserted}")

    except Exception as e:
        db.rollback()

        print("[ERROR] Failed to insert raw data.")
        print(e)

        raise e

    finally:
        db.close()


if __name__ == "__main__":
    load_raw_data(
        csv_path="data/cc_fraud_train.csv",
        data_split="train",
        data_source="kaggle",
        chunk_size=10000
    )

    load_raw_data(
        csv_path="data/cc_fraud_test.csv",
        data_split="test",
        data_source="kaggle",
        chunk_size=10000
    )