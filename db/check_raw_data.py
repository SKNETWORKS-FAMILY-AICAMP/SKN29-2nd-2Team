# db/check_raw_data.py

from sqlalchemy import text

from db.database import SessionLocal


def check_raw_data():
    """
    raw_transactions 테이블 적재 상태 확인
    """

    db = SessionLocal()

    try:
        queries = {
            "전체 데이터 수": """
                SELECT COUNT(*) AS count
                FROM raw_transactions
            """,

            "train/test별 데이터 수": """
                SELECT data_split, COUNT(*) AS count
                FROM raw_transactions
                GROUP BY data_split
            """,

            "출처별 데이터 수": """
                SELECT data_source, COUNT(*) AS count
                FROM raw_transactions
                GROUP BY data_source
            """,

            "train/test별 사기 거래 수": """
                SELECT 
                    data_split,
                    is_fraud,
                    COUNT(*) AS count
                FROM raw_transactions
                GROUP BY data_split, is_fraud
                ORDER BY data_split, is_fraud
            """,

            "샘플 데이터 5개": """
                SELECT 
                    txn_id,
                    data_split,
                    data_source,
                    trans_date_trans_time,
                    merchant,
                    category,
                    amt,
                    state,
                    is_fraud
                FROM raw_transactions
                LIMIT 5
            """
        }

        for title, query in queries.items():
            print(f"\n========== {title} ==========")

            result = db.execute(text(query))

            for row in result:
                print(dict(row._mapping))

    except Exception as e:
        print("[ERROR] Failed to check raw data.")
        print(e)

    finally:
        db.close()


if __name__ == "__main__":
    check_raw_data()
