from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.stock_service import StockService
from utils.indicators import calculate_atr


@dataclass
class TradeSignalService:
    stock: StockService | None = None

    def __post_init__(self):
        if self.stock is None:
            self.stock = StockService()

    def _map_timeframe(self, timeframe: str) -> Tuple[str, str]:
        tf = timeframe.lower()
        if tf == "intraday":
            return ("7d", "5m")
        if tf == "monthly":
            return ("3y", "1wk")
        return ("6mo", "1d")

    def generate_trade_signal(
        self,
        symbol: str,
        timeframe: str = "swing",
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        conservative_risk_reward: float = 1.5,
        aggressive_risk_reward: float = 3.0,
    ) -> Dict[str, Any]:
        """Generate a simple rule-based trade signal.

        Returns a dict with: symbol, timeframe, signal, entry_price, stop_price, targets,
        exit_rule, confidence, reasons, timestamp_utc
        """
        period, interval = self._map_timeframe(timeframe)
        history = self.stock.get_history(symbol, period=period, interval=interval)
        if history is None or history.empty:
            return {"error": "Insufficient historical data to generate signals.", "symbol": symbol}

        # Ensure indicators exist (stock.get_history already adds indicators); compute ATR
        try:
            latest = history.iloc[-1]
        except Exception:
            return {"error": "Insufficient historical data to generate signals.", "symbol": symbol}

        close = float(latest.get("Close", 0.0))
        sma20 = float(latest.get("SMA_20") or 0.0)
        sma50 = float(latest.get("SMA_50") or 0.0)
        rsi = float(latest.get("RSI") or 0.0)

        atr_series = calculate_atr(history, period=atr_period)
        atr_latest = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0

        reasons: List[str] = []
        signal = "neutral"

        sma_agreement_buy = close > sma20 and sma20 > sma50
        sma_agreement_sell = close < sma20 and sma20 < sma50

        if sma_agreement_buy:
            signal = "buy"
            reasons.append("Trend: price > SMA20 > SMA50")
        elif sma_agreement_sell:
            signal = "sell"
            reasons.append("Trend: price < SMA20 < SMA50")
        else:
            reasons.append("No clear SMA trend agreement")

        if 30 < rsi < 70:
            reasons.append(f"RSI supportive: {rsi:.1f}")
            rsi_support = True
        else:
            rsi_support = False
            reasons.append(f"RSI: {rsi:.1f}")

        entry_price: Optional[float] = None
        stop_price: Optional[float] = None
        targets: List[float] = []
        exit_rule: Optional[str] = None

        if signal in ("buy", "sell") and atr_latest > 0:
            entry_price = close
            if signal == "buy":
                stop_price = max(0.0, entry_price - atr_multiplier * atr_latest)
                risk = entry_price - stop_price
                targets = [round(entry_price + conservative_risk_reward * risk, 4), round(entry_price + aggressive_risk_reward * risk, 4)]
                exit_rule = "Hit stop / Hit target / SMA20 crosses below SMA50"
            else:
                stop_price = entry_price + atr_multiplier * atr_latest
                risk = stop_price - entry_price
                targets = [round(entry_price - conservative_risk_reward * risk, 4), round(entry_price - aggressive_risk_reward * risk, 4)]
                exit_rule = "Hit stop / Hit target / SMA20 crosses above SMA50"
            reasons.append(f"ATR {atr_latest:.4f} used for stop calculation")
        else:
            reasons.append("No ATR-based entry computed")

        # Confidence scoring (deterministic, simple)
        confidence = 0.5
        if sma_agreement_buy or sma_agreement_sell:
            confidence += 0.2
        if rsi_support:
            confidence += 0.1
        if atr_latest > 0:
            confidence += 0.1
        if atr_latest > 0 and atr_latest / max(1.0, close) > 0.05:
            confidence -= 0.15
            reasons.append("High ATR relative to price reduces confidence")
        confidence = max(0.0, min(1.0, confidence))

        out = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "entry_price": None if entry_price is None else round(float(entry_price), 4),
            "stop_price": None if stop_price is None else round(float(stop_price), 4),
            "targets": targets,
            "exit_rule": exit_rule,
            "confidence": round(float(confidence), 3),
            "reasons": reasons,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        return out
