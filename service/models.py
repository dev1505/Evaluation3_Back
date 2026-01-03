import uuid
from sqlalchemy import UUID, Column, DateTime, Float, Integer, String, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class UserDocs(Base):
    __tablename__ = "eval_user_docs"
    __table_args__ = {"schema": "public"}
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    size_bytes = Column(Float, nullable=False)
    size_kb = Column(Float, nullable=False)
    size_mb = Column(Float, nullable=False)
    uploaded_at = Column(String, nullable=False)
    extension = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)


# class UserDocs(Base):
#     __tablename__ = "eval_user_docs"
#     __table_args__ = {"schema": "public"}
#     filename = Column(String, nullable=True)
#     size_bytes = Column(Float, nullable=False)
#     size_kb = Column(Float, nullable=False)
#     size_mb = Column(Float, nullable=False)
#     extension = Column(String, nullable=False)
#     mime_type = Column(String, nullable=False)
#     uploaded_at = Column(String, nullable=False)
