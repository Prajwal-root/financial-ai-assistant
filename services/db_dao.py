from database.db import SessionLocal
from database.models import TradeRecommendation
import json


def log_trade_recommendation(symbol: str, timeframe: str, recommendation: dict, acknowledged: int = 0) -> int:
    """Insert a trade recommendation record and return the record id."""
    db = SessionLocal()
    try:
        rec = TradeRecommendation(symbol=symbol, timeframe=timeframe, recommendation=recommendation, acknowledged=acknowledged)
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id
    finally:
        db.close()
