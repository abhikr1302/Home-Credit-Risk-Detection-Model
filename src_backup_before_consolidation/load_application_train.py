import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, inspect, text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
TABLE_NAME = "application_train"
SCHEMA_NAME = "raw"
READ_CHUNK_SIZE = 10_000

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB"),
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
)


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return dataframe


def load_application_train() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH.resolve()}"
        )

    print(f"Loading {CSV_PATH}")
    print(f"Destination: {SCHEMA_NAME}.{TABLE_NAME}")

    start_time = time.time()
    total_rows = 0
    first_chunk = True

    csv_chunks = pd.read_csv(
        CSV_PATH,
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, dataframe in enumerate(
        csv_chunks,
        start=1,
    ):
        dataframe = standardize_columns(dataframe)

        dataframe.to_sql(
            name=TABLE_NAME,
            con=engine,
            schema=SCHEMA_NAME,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=1_000,
        )

        first_chunk = False
        total_rows += len(dataframe)

        print(
            f"Chunk {chunk_number} completed | "
            f"Rows loaded: {total_rows:,}"
        )

    elapsed_minutes = (time.time() - start_time) / 60

    with engine.connect() as connection:
        database_count = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {SCHEMA_NAME}.{TABLE_NAME}
                """
            )
        ).scalar_one()

    print("\nLoading completed.")
    print(f"CSV rows loaded: {total_rows:,}")
    print(f"Database rows: {database_count:,}")
    print(f"Time taken: {elapsed_minutes:.2f} minutes")

    if total_rows != database_count:
        raise ValueError(
            "CSV and database row counts do not match."
        )


if __name__ == "__main__":
    load_application_train()