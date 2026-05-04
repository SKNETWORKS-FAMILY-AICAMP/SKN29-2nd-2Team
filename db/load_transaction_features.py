# db/load_transaction_features.py

import math
import pandas as pd
from tqdm import tqdm

from db.database import SessionLocal
from db.models import RawTransaction, TransactionFeature


def preprocess_feature_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    processed CSV chunk 전처리 함수

    역할:
    - 컬럼명을 ORM 기준으로 변경
    - high_amt_far를 Boolean으로 변환
    - zip_code를 문자열로 변환
    - NaN을 None으로 변환
    """

    if "Unnamed: 0" in chunk.columns:
        chunk = chunk.drop(columns=["Unnamed: 0"])

    chunk = chunk.rename(columns={
        "zip": "zip_code",
        "lat": "customer_lat",
        "long": "customer_long",
        "merch_lat": "merchant_lat",
        "merch_long": "merchant_long",
    })

    chunk["zip_code"] = chunk["zip_code"].astype(str)

    if "high_amt_far" in chunk.columns:
        chunk["high_amt_far"] = chunk["high_amt_far"].astype(bool)

    chunk = chunk.where(pd.notnull(chunk), None)

    return chunk


def get_txn_ids_by_split(db, data_split: str) -> list[int]:
    """
    raw_transactions에서 특정 data_split의 txn_id 목록을 조회한다.

    주의:
    - processed CSV의 row 순서와 raw_transactions 적재 순서가 같다는 가정
    - 따라서 txn_id ASC 순서로 가져온다.
    """

    rows = (
        db.query(RawTransaction.txn_id)
        .filter(RawTransaction.data_split == data_split)
        .order_by(RawTransaction.txn_id.asc())
        .all()
    )

    return [row.txn_id for row in rows]


def chunk_to_feature_objects(
    chunk: pd.DataFrame,
    txn_ids: list[int],
    start_index: int,
    data_split: str,   
    data_source: str
) -> list[TransactionFeature]:
    """
    processed CSV chunk를 TransactionFeature ORM 객체 리스트로 변환한다.

    역할:
    - chunk row와 raw_transactions의 txn_id를 순서대로 매칭
    - transaction_features는 txn_id를 PK/FK로 사용
    """

    objects = []

    for offset, (_, row) in enumerate(chunk.iterrows()):
        txn_id = txn_ids[start_index + offset]

        obj = TransactionFeature(
            txn_id=txn_id,

            data_split=data_split,
            data_source=data_source,

            category=row["category"],
            amt=row["amt"],
            gender=row["gender"],
            state=row["state"],
            zip_code=row["zip_code"],

            customer_lat=row["customer_lat"],
            customer_long=row["customer_long"],
            city_pop=row["city_pop"],
            unix_time=row["unix_time"],

            merchant_lat=row["merchant_lat"],
            merchant_long=row["merchant_long"],

            is_fraud=row["is_fraud"],

            trans_hour=row["trans_hour"],
            trans_dayofweek=row["trans_dayofweek"],
            trans_month=row["trans_month"],
            trans_day=row["trans_day"],

            age=row["age"],
            distance_km=row["distance_km"],
            amt_zscore=row["amt_zscore"],
            hour_dev=row["hour_dev"],
            high_amt_far=row["high_amt_far"],
        )

        objects.append(obj)

    return objects


def load_transaction_features(
    csv_path: str,
    data_split: str,
    data_source: str = "kaggle",
    chunk_size: int = 10000
):
    """
    processed CSV 데이터를 transaction_features 테이블에 적재한다.

    처리 흐름:
    1. raw_transactions에서 해당 split의 txn_id 목록 조회
    2. processed CSV를 chunk 단위로 읽기
    3. 컬럼명 정리
    4. txn_id와 row 순서 매칭
    5. transaction_features에 bulk insert
    """

    print("[INFO] Loading transaction features...")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] Data split: {data_split}")

    total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
    total_chunks = math.ceil(total_rows / chunk_size)

    db = SessionLocal()
    total_inserted = 0

    try:
        txn_ids = get_txn_ids_by_split(db, data_split)

        print(f"[INFO] Raw txn count: {len(txn_ids)}")
        print(f"[INFO] Feature CSV rows: {total_rows}")

        if len(txn_ids) != total_rows:
            raise ValueError(
                f"raw_transactions count와 feature CSV row 수가 다릅니다. "
                f"raw={len(txn_ids)}, csv={total_rows}"
            )

        chunk_reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            index_col=0
        )

        progress_bar = tqdm(
            chunk_reader,
            total=total_chunks,
            desc=f"Loading {data_split} features"
        )

        current_index = 0

        for chunk in progress_bar:
            chunk = preprocess_feature_chunk(chunk)

            objects = chunk_to_feature_objects(
                chunk=chunk,
                txn_ids=txn_ids,
                start_index=current_index,
                data_split=data_split,
                data_source=data_source
            )

            db.bulk_save_objects(objects)
            db.commit()

            total_inserted += len(objects)
            current_index += len(objects)

            progress_bar.set_postfix({
                "Inserted": total_inserted
            })

        print("[INFO] Transaction feature insert complete.")
        print(f"[INFO] Total inserted rows: {total_inserted}")

    except Exception as e:
        db.rollback()
        print("[ERROR] Failed to insert transaction features.")
        print(e)
        raise e

    finally:
        db.close()


if __name__ == "__main__":
    load_transaction_features(
        csv_path="data/cc_fraud_train_processed.csv",
        data_split="train",
        data_source="kaggle",
        chunk_size=10000
    )

    load_transaction_features(
        csv_path="data/cc_fraud_test_processed.csv",
        data_split="test",
        data_source="kaggle",
        chunk_size=10000
    )