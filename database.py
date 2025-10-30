# --- Import required libraries ---
from databases import Database        # for async DB operations
import sqlalchemy                     # for schema and SQL handling
import os                             # to read environment variables
from model import metadata            # import table metadata (schema)

# --- Read environment variables from Docker ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_NAME = os.getenv("DB_NAME", "dust_detection")

# --- Build the full database URL ---
# Example: postgresql+asyncpg://postgres:admin@db:5432/dust_detection
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Create async Database instance ---
database = Database(DATABASE_URL)

# --- Create engine for table creation (sync only) ---
# SQLAlchemy’s `create_all()` can’t use async engines, so we replace "+asyncpg" with "".
engine = sqlalchemy.create_engine(DATABASE_URL.replace("+asyncpg", ""))

# --- Create all tables (if not already existing) ---
metadata.create_all(engine)