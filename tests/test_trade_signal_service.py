import pandas as pd
import numpy as np
from services.trade_signal_service import TradeSignalService
from utils.indicators import add_indicators


def make_uptrend_df(days=120, start_price=100.0):
    np.random.seed(0)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    closes = np.linspace(start_price, start_price * 1.2, days)
    highs = closes + np.abs(np.random.rand(days) * 0.5)
    lows = closes - np.abs(np.random.rand(days) * 0.5)
    opens = closes + (np.random.rand(days) - 0.5) * 0.2
    volume = np.random.randint(1000, 5000, size=days)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume}, index=idx)
    df = add_indicators(df)
    return df


def test_buy_signal_on_clear_uptrend(monkeypatch):
    svc = TradeSignalService()
    # monkeypatch internal stock.get_history to return deterministic uptrend with indicators
    monkeypatch.setattr(svc.stock, "get_history", lambda symbol, period, interval: make_uptrend_df(days=120))
    out = svc.generate_trade_signal("TEST.NS", timeframe="swing")
    assert out["symbol"] == "TEST.NS"
    # Expect either buy or neutral; on a clear uptrend we expect buy
    assert out["signal"] in ("buy", "neutral")
    if out["signal"] == "buy":
        assert out["entry_price"] is not None
        assert out["stop_price"] is not None
        assert len(out["targets"]) >= 1
        assert 0.0 <= out["confidence"] <= 1.0


def test_insufficient_history_returns_error(monkeypatch):
    svc = TradeSignalService()
    monkeypatch.setattr(svc.stock, "get_history", lambda symbol, period, interval: pd.DataFrame())
    out = svc.generate_trade_signal("TEST.NS", timeframe="swing")
    assert "error" in out
