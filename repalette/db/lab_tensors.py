import datetime
from sqlalchemy import Column, Integer, String, DateTime

from repalette.db import Base


class LABTensor(Base):
    __tablename__ = "lab_tensors"
    id = Column("id", Integer, primary_key=True)
    image_path = Column("image_path", String)
    palette_path = Column("palette_path", String)
    url = Column("url", String, unique=True)
    name = Column("name", String, default="")
    height = Column("height", Integer, default=0)
    width = Column("width", Integer, default=0)
    created_at = Column("created_at", DateTime, default=datetime.datetime.utcnow)
