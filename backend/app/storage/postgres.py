import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://vision:vision_pass@localhost:5432/vision_db")

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TrajectoryLog(Base):
    __tablename__ = "trajectory_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    track_id = Column(Integer, index=True)
    object_type = Column(String)
    timestamp = Column(DateTime, index=True)
    pos_x = Column(Integer)
    pos_y = Column(Integer)
    metadata_json = Column(JSON, nullable=True)
