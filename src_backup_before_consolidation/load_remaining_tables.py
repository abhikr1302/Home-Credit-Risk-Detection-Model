import csv
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DATABASE_SCHEMA = "raw"
READ_CHUNK_SIZE = 50_000

FILE_TABLE_MAPPING = {
    "application_test.csv": "application_test",
    "bureau.csv": "bureau",
    "bureau_balance.csv": "bureau_balance",
    "previous_application.csv": "previous_application",
    "installments_payments.csv": "installments_payments",
    "credit_card_balance.csv": "credit_card_balance",
    "POS_CASH_balance.csv": "pos_cash_balance",
}

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


def postgresql_copy(table, connection, columns, data_iterator):
    """
    Use PostgreSQL COPY instead of individual INSERT statements.
    This is substantially faster for large CSV files.
    """
    buffer = StringIO()
    writer = csv.writer(buffer)

    writer.writerows(data_iterator)
    buffer.seek(0)

    schema_name = table.schema or "public"
    column_names = ", ".join(
        f'"{column}"' for column in columns
    )

    copy_statement = (
        f'COPY "{schema_name}"."{table.name}" '
        f"({column_names}) "
        f"FROM STDIN WITH (FORMAT CSV)"
    )

    raw_connection = connection.connection

    with raw_connection.cursor() as cursor:
        cursor.copy_expert(
            sql=copy_statement,
            file=buffer,
        )


def verify_database_connection() -> None:
    with engine.connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT
                    current_database(),
                    current_user
                """
            )
        ).one()

    print(f"Database: {result[0]}")
    print(f"User: {result[1]}")


def load_csv_to_postgresql(
    file_name: str,
    destination_table: str,
) -> None:
    file_path = RAW_DATA_DIRECTORY / file_name
    staging_table = f"{destination_table}_staging"

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path.resolve()}"
        )

    print("\n" + "=" * 70)
    print(f"Source: {file_path}")
    print(
        f"Destination: "
        f"{DATABASE_SCHEMA}.{destination_table}"
    )
    print("=" * 70)

    start_time = time.time()
    total_rows = 0
    first_chunk = True

    csv_chunks = pd.read_csv(
        file_path,
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, dataframe in enumerate(
        csv_chunks,
        start=1,
    ):
        dataframe = standardize_columns(dataframe)

        dataframe.to_sql(
            name=staging_table,
            con=engine,
            schema=DATABASE_SCHEMA,
            if_exists="replace" if first_chunk else "append",
            index=False,
            method=postgresql_copy,
        )

        first_chunk = False
        total_rows += len(dataframe)

        print(
            f"Chunk {chunk_number:,} completed | "
            f"Total rows: {total_rows:,}"
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                DROP TABLE IF EXISTS
                    {DATABASE_SCHEMA}.{destination_table};

                ALTER TABLE
                    {DATABASE_SCHEMA}.{staging_table}
                RENAME TO
                    {destination_table};
                """
            )
        )

    with engine.connect() as connection:
        database_rows = connection.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {DATABASE_SCHEMA}.{destination_table}
                """
            )
        ).scalar_one()

    elapsed_minutes = (time.time() - start_time) / 60

    print(f"CSV rows: {total_rows:,}")
    print(f"Database rows: {database_rows:,}")
    print(f"Time taken: {elapsed_minutes:.2f} minutes")

    if total_rows != database_rows:
        raise ValueError(
            f"Row-count mismatch for {destination_table}"
        )

    print(f"{destination_table} loaded successfully.")


def main() -> None:
    verify_database_connection()

    for file_name, table_name in FILE_TABLE_MAPPING.items():
        load_csv_to_postgresql(
            file_name=file_name,
            destination_table=table_name,
        )

    print("\nAll remaining tables loaded successfully.")


if __name__ == "__main__":
    main()