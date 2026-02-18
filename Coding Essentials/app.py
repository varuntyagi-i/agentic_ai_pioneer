from __future__ import annotations

from datetime import datetime
from math import sqrt
from typing import Any

import yfinance as yf
from flask import Flask, jsonify, request

app = Flask(__name__)


def _error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _parse_json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _normalize_symbol(symbol: Any) -> str:
    if symbol is None:
        return ""
    return str(symbol).strip().upper()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_to_records(df):
    records = []
    for idx, row in df.iterrows():
        timestamp = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        records.append(
            {
                "timestamp": timestamp,
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": int(row.get("Volume")) if row.get("Volume") is not None else None,
            }
        )
    return records


def _compute_rsi(close_series, period: int = 14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@app.get("/company-info")
def company_info():
    symbol = _normalize_symbol(request.args.get("symbol"))
    if not symbol:
        return _error("Missing required query parameter: symbol")

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    if not info:
        return _error(f"No company info found for symbol '{symbol}'", 404)

    officers = info.get("companyOfficers") or []
    key_officers = []
    for officer in officers:
        if not isinstance(officer, dict):
            continue
        name = officer.get("name")
        title = officer.get("title")
        if name or title:
            key_officers.append({"name": name, "title": title})

    response = {
        "symbol": symbol,
        "Company Name": info.get("longName") or info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "Business Summary": info.get("longBusinessSummary"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "employees": info.get("fullTimeEmployees"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "key_officers": key_officers,
    }
    return jsonify(response)


@app.get("/stock-market-data")
def stock_market_data():
    symbol = _normalize_symbol(request.args.get("symbol"))
    if not symbol:
        return _error("Missing required query parameter: symbol")

    ticker = yf.Ticker(symbol)
    fast = ticker.fast_info or {}

    response = {
        "symbol": symbol,
        "currency": fast.get("currency"),
        "exchange": fast.get("exchange"),
        "current_market_price": _safe_float(fast.get("lastPrice")),
        "previous_close": _safe_float(fast.get("previousClose")),
        "open": _safe_float(fast.get("open")),
        "day_high": _safe_float(fast.get("dayHigh")),
        "day_low": _safe_float(fast.get("dayLow")),
        "volume": fast.get("lastVolume"),
        "market_cap": fast.get("marketCap"),
        "fifty_day_average": _safe_float(fast.get("fiftyDayAverage")),
        "two_hundred_day_average": _safe_float(fast.get("twoHundredDayAverage")),
        "as_of": datetime.utcnow().isoformat() + "Z",
    }

    if response["current_market_price"] is None:
        intraday = ticker.history(period="1d", interval="1m")
        if intraday.empty:
            return _error(f"No market data found for symbol '{symbol}'", 404)
        latest = intraday.iloc[-1]
        response.update(
            {
                "current_market_price": _safe_float(latest.get("Close")),
                "last_price": _safe_float(latest.get("Close")),
                "open": _safe_float(latest.get("Open")),
                "day_high": _safe_float(intraday["High"].max()),
                "day_low": _safe_float(intraday["Low"].min()),
                "volume": int(intraday["Volume"].sum()) if "Volume" in intraday else None,
            }
        )

    return jsonify(response)


@app.post("/historical-market-data")
def historical_market_data():
    payload = _parse_json_payload()
    symbol = _normalize_symbol(payload.get("symbol"))
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    interval = payload.get("interval", "1d")

    if not symbol:
        return _error("Missing required field in JSON body: symbol")
    if not start_date or not end_date:
        return _error("Missing required fields in JSON body: start_date, end_date")

    try:
        start = datetime.fromisoformat(str(start_date))
        end = datetime.fromisoformat(str(end_date))
    except ValueError:
        return _error("Dates must be in ISO format, e.g. 2025-01-01")

    if start >= end:
        return _error("start_date must be earlier than end_date")

    df = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
    if df.empty:
        return _error("No historical data found for given symbol/date range", 404)

    return jsonify(
        {
            "symbol": symbol,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "interval": interval,
            "rows": len(df),
            "data": _history_to_records(df),
        }
    )


@app.post("/analytical-insights")
def analytical_insights():
    payload = _parse_json_payload()
    symbol = _normalize_symbol(payload.get("symbol"))
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    interval = payload.get("interval", "1d")

    if not symbol:
        return _error("Missing required field in JSON body: symbol")
    if not start_date or not end_date:
        return _error("Missing required fields in JSON body: start_date, end_date")

    try:
        start = datetime.fromisoformat(str(start_date))
        end = datetime.fromisoformat(str(end_date))
    except ValueError:
        return _error("Dates must be in ISO format, e.g. 2025-01-01")

    if start >= end:
        return _error("start_date must be earlier than end_date")

    df = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
    if df.empty or "Close" not in df:
        return _error("No historical data available for analysis", 404)

    close = df["Close"].dropna()
    if close.empty:
        return _error("Close price data is unavailable for analysis", 404)

    returns = close.pct_change().dropna()
    total_return_pct = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 1 else 0.0
    volatility_pct = returns.std() * sqrt(252) * 100 if not returns.empty else 0.0

    ma20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else None
    ma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
    rsi = _compute_rsi(close).iloc[-1] if len(close) >= 15 else None

    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min() * 100) if not drawdown.empty else 0.0

    trend = "neutral"
    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            trend = "bullish"
        elif ma20 < ma50:
            trend = "bearish"

    insights = []
    if trend == "bullish":
        insights.append("Short-term trend is bullish (20-day average is above 50-day average).")
    elif trend == "bearish":
        insights.append("Short-term trend is bearish (20-day average is below 50-day average).")
    else:
        insights.append("Trend signal is neutral due to limited data or mixed moving-average behavior.")

    if rsi is not None:
        if rsi > 70:
            insights.append("RSI indicates overbought conditions; expect possible pullback risk.")
        elif rsi < 30:
            insights.append("RSI indicates oversold conditions; rebound potential may exist.")
        else:
            insights.append("RSI is in a balanced zone, suggesting no extreme momentum conditions.")

    if volatility_pct > 40:
        insights.append("Volatility is high; position sizing and risk controls are important.")
    elif volatility_pct < 20:
        insights.append("Volatility is relatively low; price movement has been comparatively stable.")

    if total_return_pct > 0:
        insights.append("The analyzed period shows positive returns overall.")
    elif total_return_pct < 0:
        insights.append("The analyzed period shows negative returns overall.")

    action = "hold"
    if trend == "bullish" and (rsi is None or rsi < 70):
        action = "buy_bias"
    elif trend == "bearish" and (rsi is None or rsi > 30):
        action = "sell_bias"

    return jsonify(
        {
            "symbol": symbol,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "interval": interval,
            "analysis": {
                "latest_close": _safe_float(close.iloc[-1]),
                "total_return_pct": round(total_return_pct, 4),
                "annualized_volatility_pct": round(volatility_pct, 4),
                "max_drawdown_pct": round(max_drawdown_pct, 4),
                "moving_average_20": _safe_float(ma20),
                "moving_average_50": _safe_float(ma50),
                "rsi_14": _safe_float(rsi),
                "trend": trend,
                "action_signal": action,
            },
            "insights": insights,
            "disclaimer": "Analysis is informational only and not financial advice.",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
