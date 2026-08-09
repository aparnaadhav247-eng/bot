"""
Thin wrapper around Finnhub's REST API.

Each function here maps 1:1 to a Claude tool defined in ai.py. Keep these
functions returning small, clean dicts (not raw API dumps) — Claude reasons
over what you give it, so noisy payloads produce noisy answers.

Finnhub free tier docs: https://finnhub.io/docs/api
"""
import os
import requests

BASE_URL = "https://finnhub.io/api/v1"


def _key():
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set in .env")
    return key


def _get(path: str, params: dict):
    params = {**params, "token": _key()}
    resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_quote(ticker: str) -> dict:
    """Real-time quote: current price, change, high/low/open, prev close."""
    data = _get("quote", {"symbol": ticker.upper()})
    if not data or data.get("c") is None:
        return {"error": f"No quote data found for {ticker}"}
    return {
        "ticker": ticker.upper(),
        "current_price": data.get("c"),
        "change": data.get("d"),
        "percent_change": data.get("dp"),
        "day_high": data.get("h"),
        "day_low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
    }


def get_company_profile(ticker: str) -> dict:
    """Basic company info: name, industry, market cap, exchange, IPO date."""
    data = _get("stock/profile2", {"symbol": ticker.upper()})
    if not data:
        return {"error": f"No profile found for {ticker}"}
    return {
        "ticker": ticker.upper(),
        "name": data.get("name"),
        "industry": data.get("finnhubIndustry"),
        "market_cap_musd": data.get("marketCapitalization"),
        "exchange": data.get("exchange"),
        "ipo_date": data.get("ipo"),
        "website": data.get("weburl"),
        "country": data.get("country"),
    }


def get_company_news(ticker: str, from_date: str, to_date: str, limit: int = 8) -> list:
    """
    Recent news for a company. Dates must be YYYY-MM-DD.
    Returns a trimmed list (headline, source, url, datetime, summary).
    """
    data = _get("company-news", {"symbol": ticker.upper(), "from": from_date, "to": to_date})
    trimmed = []
    for item in data[:limit]:
        trimmed.append({
            "headline": item.get("headline"),
            "source": item.get("source"),
            "url": item.get("url"),
            "datetime": item.get("datetime"),
            "summary": (item.get("summary") or "")[:400],
        })
    return trimmed


def get_earnings_calendar(from_date: str, to_date: str, ticker: str = None) -> list:
    """Upcoming/recent earnings dates + EPS estimates. Dates must be YYYY-MM-DD."""
    params = {"from": from_date, "to": to_date}
    if ticker:
        params["symbol"] = ticker.upper()
    data = _get("calendar/earnings", params)
    events = data.get("earningsCalendar", [])
    return [{
        "ticker": e.get("symbol"),
        "date": e.get("date"),
        "eps_estimate": e.get("epsEstimate"),
        "eps_actual": e.get("epsActual"),
        "revenue_estimate": e.get("revenueEstimate"),
        "hour": e.get("hour"),  # bmo/amc = before/after market
    } for e in events]


def get_peers(ticker: str) -> list:
    """Peer/competitor tickers in the same sector, for comparisons."""
    return _get("stock/peers", {"symbol": ticker.upper()})


def get_basic_financials(ticker: str) -> dict:
    """Key valuation/financial ratios: P/E, margins, growth, 52wk range, etc."""
    data = _get("stock/metric", {"symbol": ticker.upper(), "metric": "all"})
    m = data.get("metric", {})
    return {
        "ticker": ticker.upper(),
        "pe_ttm": m.get("peTTM"),
        "eps_ttm": m.get("epsTTM"),
        "revenue_growth_yoy": m.get("revenueGrowthTTMYoy"),
        "gross_margin": m.get("grossMarginTTM"),
        "net_margin": m.get("netProfitMarginTTM"),
        "52wk_high": m.get("52WeekHigh"),
        "52wk_low": m.get("52WeekLow"),
        "beta": m.get("beta"),
        "dividend_yield": m.get("dividendYieldIndicatedAnnual"),
    }
