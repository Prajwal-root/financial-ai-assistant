from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, JSON
from database.db import Base


class TradeRecommendation(Base):
    __tablename__ = "trade_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(32), nullable=False)
    recommendation = Column(JSON, nullable=False)
    acknowledged = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
