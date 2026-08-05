"""Pure parsers for live next-draw jackpot sources (no HTTP, no DB).

Kept free of I/O so the parsing is unit-testable against saved fixtures;
lottery_service.fetch_live_jackpot owns the HTTP + caching.

Source quirks these encode (verified from the prod VM, 2026-08-05):
- powerball.com's Drupal JSON API serves the SPA HTML homepage to
  non-browser clients regardless of ?_format=json — but that homepage
  carries the advertised jackpot and cash value in static markup
  ("Estimated Jackpot" / "Cash Value" labels, each followed by a
  game-jackpot-number span). Scraping the page we reliably receive beats
  fighting the CDN for the API we don't. Requests must send
  Accept-Encoding: gzip — the default negotiation yields brotli, which
  httpx can't decode without the optional brotli package.
- megamillions.com's ASMX endpoint returns XML-wrapping-JSON:
  <string xmlns="...">{"Jackpot": {"NextPrizePool": ..., ...}}</string>.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

JACKPOT_SOURCES = {
    "powerball": "https://www.powerball.com/api/v1/estimates/powerball?_format=json",
    "powerball_fallback_page": "https://www.powerball.com/",
    "mega_millions": "https://www.megamillions.com/cmspages/utilservice.asmx/GetLatestDrawData",
}

# Browser-ish UA (several of these endpoints 500 or block on default python
# fingerprints) + gzip-only so httpx can always decode the body.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Encoding": "gzip",
}


def parse_dollar_amount(s: str) -> Optional[float]:
    """'$786 Million' / '$1.2 Billion' / '$341,600,000' -> dollars."""
    m = re.search(r"\$?\s*([\d,.]+)\s*(billion|million)?", s, re.I)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit == "billion":
        value *= 1e9
    elif unit == "million":
        value *= 1e6
    return value


def parse_powerball_api(body: str) -> Optional[Dict[str, float]]:
    """The Drupal estimates API's real JSON (a list of nodes with
    field_prize_amount / field_prize_amount_cash). Returns None when the CDN
    served HTML instead — caller falls back to the homepage scrape."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    node = data[0] if isinstance(data, list) and data else data
    if not isinstance(node, dict):
        return None
    adv = parse_dollar_amount(str(node.get("field_prize_amount", "")))
    cash = parse_dollar_amount(str(node.get("field_prize_amount_cash", "")))
    if not adv:
        return None
    return {"advertised": adv, "cash_value": cash or 0.0}


def parse_powerball_homepage(html: str) -> Optional[Dict[str, float]]:
    """Advertised + cash value from the powerball.com homepage markup:
    a label ("Estimated Jackpot" / "Cash Value") followed within a couple of
    spans by <span class="game-jackpot-number ...">$786 Million</span>."""

    def value_after(label: str) -> Optional[float]:
        i = html.find(label)
        if i == -1:
            return None
        m = re.search(
            r"game-jackpot-number[^>]*>\s*(\$[\d,.]+\s*(?:Million|Billion)?)",
            html[i : i + 2000],
            re.I,
        )
        return parse_dollar_amount(m.group(1)) if m else None

    advertised = value_after("Estimated Jackpot")
    if not advertised:
        return None
    cash = value_after("Cash Value")
    return {"advertised": advertised, "cash_value": cash or 0.0}


def parse_megamillions_payload(body: str) -> Optional[Dict[str, float]]:
    """XML-wrapped JSON from GetLatestDrawData -> next-draw jackpot."""
    inner = re.search(r">(\{.*\})<", body, re.S)
    if not inner:
        return None
    try:
        payload = json.loads(inner.group(1))
    except json.JSONDecodeError:
        return None
    jack = payload.get("Jackpot") or {}
    adv = jack.get("NextPrizePool")
    if not adv:
        return None
    return {"advertised": float(adv), "cash_value": float(jack.get("NextCashValue") or 0.0)}
