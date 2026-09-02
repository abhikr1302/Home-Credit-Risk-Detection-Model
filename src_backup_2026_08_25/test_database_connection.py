import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

load_dotenv()

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

with engine.connect() as connection:
    result = connection.execute(
        text(
            """
            SELECT
                current_database() AS database_name,
                current_user AS user_name
            """
        )
    ).mappings().one()

print(f"Database: {result['database_name']}")
print(f"User: {result['user_name']}")
print("Connection successful.")