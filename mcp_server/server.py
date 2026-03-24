"""
FastMCP Server – exposes financial analysis tools and generic utility tools.

Run standalone:  python -m mcp_server.server
Or import the `mcp` object for use as an ADK tool.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP(
    name="FinancialToolsServer",
    instructions=(
        "You are a financial tools server. You provide utilities for "
        "stock lookups, financial ratio calculations, SEC filing search, "
        "and general-purpose helpers like a calculator and date tools."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def calculate_financial_ratios(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_liabilities: float,
    total_equity: float,
    current_assets: float | None = None,
    current_liabilities: float | None = None,
) -> dict:
    """
    Calculate key financial ratios from basic financial statement inputs.

    Args:
        revenue: Total revenue / sales.
        net_income: Net income (profit after tax).
        total_assets: Total assets on the balance sheet.
        total_liabilities: Total liabilities.
        total_equity: Total shareholders' equity.
        current_assets: Current assets (optional, for liquidity ratios).
        current_liabilities: Current liabilities (optional, for liquidity ratios).

    Returns:
        A dictionary of computed financial ratios.
    """
    ratios: dict[str, float | str] = {}

    # Profitability
    ratios["profit_margin"] = round(net_income / revenue, 4) if revenue else "N/A"
    ratios["return_on_assets"] = round(net_income / total_assets, 4) if total_assets else "N/A"
    ratios["return_on_equity"] = round(net_income / total_equity, 4) if total_equity else "N/A"

    # Leverage
    ratios["debt_to_equity"] = (
        round(total_liabilities / total_equity, 4) if total_equity else "N/A"
    )
    ratios["debt_to_assets"] = (
        round(total_liabilities / total_assets, 4) if total_assets else "N/A"
    )

    # Liquidity (if data provided)
    if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
        ratios["current_ratio"] = round(current_assets / current_liabilities, 4)
    else:
        ratios["current_ratio"] = "N/A (missing current assets/liabilities)"

    return ratios


@mcp.tool()
def stock_price_lookup(ticker: str) -> dict:
    """
    Look up the latest stock price for a given ticker symbol.
    (Simulated – replace with a real API like Alpha Vantage or Yahoo Finance.)

    Args:
        ticker: Stock ticker symbol (e.g., "NFLX", "AAPL").

    Returns:
        Dict with ticker, price, currency, and timestamp.
    """
    # Simulated prices for demo purposes
    simulated_prices = {
        "NFLX": 1045.32,
        "AAPL": 248.15,
        "GOOGL": 192.87,
        "MSFT": 455.60,
        "AMZN": 225.40,
        "TSLA": 275.90,
        "META": 620.15,
    }

    ticker_upper = ticker.upper()
    price = simulated_prices.get(ticker_upper)

    if price is None:
        return {
            "ticker": ticker_upper,
            "error": f"Ticker '{ticker_upper}' not found in simulated data. "
            "In production, this would call a real market-data API.",
        }

    return {
        "ticker": ticker_upper,
        "price": price,
        "currency": "USD",
        "timestamp": datetime.now().isoformat(),
        "note": "Simulated price – replace with live API in production.",
    }


@mcp.tool()
def sec_filing_search(
    company_name: str,
    filing_type: str = "10-K",
) -> dict:
    """
    Search for SEC filings by company name and filing type.
    (Simulated – replace with EDGAR API integration.)

    Args:
        company_name: Name of the company (e.g., "Netflix").
        filing_type: SEC form type (e.g., "10-K", "10-Q", "4", "8-K").

    Returns:
        Dict with filing metadata and a direct EDGAR link.
    """
    # Simulated lookup
    filings_db = {
        "netflix": {
            "10-K": {
                "company": "Netflix, Inc.",
                "cik": "0001065280",
                "filing_type": "10-K",
                "filed_date": "2026-01-30",
                "period": "2025-12-31",
                "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001065280&type=10-K",
            },
            "4": {
                "company": "Netflix, Inc.",
                "cik": "0001065280",
                "filing_type": "4",
                "filed_date": "2026-03-03",
                "filer": "Sweeney Anne M",
                "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001065280&type=4",
            },
        }
    }

    company_key = company_name.lower().strip()
    company_filings = filings_db.get(company_key, {})
    filing = company_filings.get(filing_type)

    if filing:
        return filing
    return {
        "company": company_name,
        "filing_type": filing_type,
        "error": "Filing not found in simulated data. In production, query EDGAR FULL-TEXT search.",
    }


@mcp.tool()
def compare_financials(
    company_a: str,
    company_b: str,
    metric: str = "revenue",
) -> dict:
    """
    Compare a financial metric between two companies.
    (Simulated – replace with a real financial data API.)

    Args:
        company_a: First company name.
        company_b: Second company name.
        metric: Metric to compare (revenue, net_income, market_cap, employees).

    Returns:
        Comparison dict with values for both companies.
    """
    data = {
        "netflix": {"revenue": 39_000_000_000, "net_income": 8_700_000_000, "market_cap": 450_000_000_000, "employees": 16_000},
        "apple": {"revenue": 383_000_000_000, "net_income": 94_000_000_000, "market_cap": 3_500_000_000_000, "employees": 161_000},
        "google": {"revenue": 340_000_000_000, "net_income": 86_000_000_000, "market_cap": 2_100_000_000_000, "employees": 182_000},
        "microsoft": {"revenue": 245_000_000_000, "net_income": 88_000_000_000, "market_cap": 3_200_000_000_000, "employees": 221_000},
    }

    a_key = company_a.lower().strip()
    b_key = company_b.lower().strip()
    metric_key = metric.lower().strip()

    a_val = data.get(a_key, {}).get(metric_key)
    b_val = data.get(b_key, {}).get(metric_key)

    return {
        "metric": metric_key,
        company_a: a_val if a_val else "Not found",
        company_b: b_val if b_val else "Not found",
        "note": "Simulated data – replace with live financial API.",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GENERIC UTILITY TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def calculator(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: A math expression string (e.g., "2 * 3 + 4", "sqrt(144)", "1e6 / 12").

    Returns:
        Dict with the expression and its result.
    """
    # Provide a safe namespace with common math functions
    safe_ns = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@mcp.tool()
def current_date_time() -> dict:
    """Return the current date and time in ISO-8601 format."""
    now = datetime.now()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


@mcp.tool()
def json_formatter(raw_json: str) -> dict:
    """
    Parse and pretty-print a JSON string.

    Args:
        raw_json: A raw JSON string to format.

    Returns:
        Dict with the parsed object or an error message.
    """
    try:
        parsed = json.loads(raw_json)
        return {"formatted": json.dumps(parsed, indent=2), "valid": True}
    except json.JSONDecodeError as exc:
        return {"error": str(exc), "valid": False}


@mcp.tool()
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str,
) -> dict:
    """
    Convert between common units (currency shorthand, distance, weight, temperature).

    Args:
        value: The numeric value to convert.
        from_unit: Source unit (e.g., "km", "miles", "kg", "lbs", "celsius", "fahrenheit", "millions", "billions").
        to_unit: Target unit.

    Returns:
        Converted value.
    """
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("millions", "billions"): lambda v: v / 1000,
        ("billions", "millions"): lambda v: v * 1000,
        ("millions", "units"): lambda v: v * 1_000_000,
        ("billions", "units"): lambda v: v * 1_000_000_000,
    }

    key = (from_unit.lower(), to_unit.lower())
    fn = conversions.get(key)
    if fn is None:
        return {"error": f"Conversion from '{from_unit}' to '{to_unit}' is not supported."}

    return {
        "original": {"value": value, "unit": from_unit},
        "converted": {"value": round(fn(value), 6), "unit": to_unit},
    }


# ── Standalone runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
