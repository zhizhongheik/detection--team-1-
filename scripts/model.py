# --- Import SQLAlchemy components ---
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData

# --- Metadata container for all database tables ---
metadata = MetaData()

# --- Define the 'new_detections' table ---
new_detections = Table(
    "new_detections",
    metadata,
    Column("id", Integer, primary_key=True),          # Unique ID
    Column("filename", String, nullable=False),       # e.g., frame_001.jpg
    Column("bbox", sqlalchemy.JSON, nullable=False),  # bounding box [x, y, w, h]
    Column("confidence", Float, nullable=False),      # model confidence
    Column("timestamp", DateTime(timezone=True), nullable=False)  # detection time
)