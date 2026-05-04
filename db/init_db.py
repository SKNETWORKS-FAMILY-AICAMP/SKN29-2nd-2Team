# db/init_db.py

from sqlalchemy import create_engine, text

from db.database import SERVER_DATABASE_URL, DB_NAME, engine, Base

# models.py를 import 해야 Base.metadata가 ORM 테이블 정보를 인식함
from db import models


def create_database():
    """
    MySQL에 프로젝트 DB가 없으면 생성한다.
    """

    server_engine = create_engine(
        SERVER_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        future=True,
    )

    with server_engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
                "DEFAULT CHARACTER SET utf8mb4 "
                "COLLATE utf8mb4_unicode_ci"
            )
        )

        conn.commit()

    print(f"[INFO] Database '{DB_NAME}' checked/created.")


def create_tables():
    """
    models.py에 정의된 ORM 클래스 기준으로 테이블을 생성한다.
    이미 존재하는 테이블은 다시 생성하지 않는다.
    """

    Base.metadata.create_all(bind=engine)

    print("[INFO] Tables checked/created.")


def drop_tables():
    """
    개발 단계에서 테이블을 전체 삭제할 때 사용한다.
    주의: 실제 데이터가 모두 삭제되므로 필요할 때만 수동 실행한다.
    """

    Base.metadata.drop_all(bind=engine)

    print("[INFO] Tables dropped.")


def init_db():
    """
    DB 생성 후 테이블을 생성한다.
    """

    create_database()
    create_tables()

    print("[INFO] Database initialization complete.")


if __name__ == "__main__":
    create_database()
    create_tables()

    # 필요할 때만 직접 주석 해제해서 사용
    # drop_tables()