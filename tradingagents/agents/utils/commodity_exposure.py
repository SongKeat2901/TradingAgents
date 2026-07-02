"""Deterministic commodity input-exposure flag (FA-101 Phase 5, §7).

Maps a name's yfinance sector/industry to a commodity input-cost exposure
level + the primary input(s). Pure classification — no financials, no fetch.
Input-cost sensitivity feeds gross-margin risk, so the Risk & Red-Flags role
cites this alongside cyclicality. Unknown/services sectors render an honest
"low" — that is the correct answer, not a data gap.
"""
from __future__ import annotations

from typing import Any

# (keyword, level, primary inputs) — matched (substring, case-insensitive)
# against "<industry> <sector>". Order matters: first match wins, so the most
# specific / highest-exposure keywords come first.
_RULES: list[tuple[str, str, str]] = [
    ("airline", "high", "jet fuel"),
    ("oil & gas", "high", "crude oil / natural gas"),
    ("oil and gas", "high", "crude oil / natural gas"),
    ("coal", "high", "thermal/coking coal"),
    ("steel", "high", "iron ore / coking coal"),
    ("aluminum", "high", "aluminium / bauxite"),
    ("copper", "high", "copper"),
    ("gold", "high", "gold ore / energy"),
    ("silver", "high", "silver ore"),
    ("chemical", "high", "oil derivatives / feedstock"),
    ("agricultural inputs", "high", "fertilizer feedstock / natural gas"),
    ("packaged foods", "high", "agricultural commodities"),
    ("farm products", "high", "agricultural commodities"),
    ("beverages", "moderate", "agricultural commodities / aluminium"),
    ("confectioners", "high", "cocoa / sugar / dairy"),
    ("restaurants", "moderate", "food commodities"),
    ("auto manufacturers", "moderate", "steel / aluminium / lithium"),
    ("auto parts", "moderate", "steel / aluminium"),
    ("building products", "moderate", "lumber / steel / cement"),
    ("construction", "moderate", "lumber / steel / cement"),
    ("paper", "moderate", "pulp / energy"),
    ("packaging", "moderate", "resins / paper / aluminium"),
    ("apparel", "moderate", "cotton / synthetics"),
    ("textile", "moderate", "cotton / synthetics"),
    ("utilities", "moderate", "fuel (often regulated pass-through)"),
    ("basic materials", "high", "raw metals / minerals"),
    ("energy", "high", "crude oil / natural gas"),
    ("industrials", "moderate", "input metals / energy"),
]


def compute_commodity_exposure(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    sector = (fin.get("sector") or "").strip()
    industry = (fin.get("industry") or "").strip()
    if not sector and not industry:
        return {"exposure": "low", "primary_inputs": None,
                "rationale": "sector/industry not classified", "classified": False}
    hay = f"{industry} {sector}".lower()
    for kw, level, inputs in _RULES:
        if kw in hay:
            return {"exposure": level, "primary_inputs": inputs,
                    "rationale": f"matched '{kw}' in {industry or sector}",
                    "classified": True}
    return {"exposure": "low", "primary_inputs": None,
            "rationale": f"{industry or sector} is not a commodity-input-intensive industry",
            "classified": True}


def format_commodity_block(result: dict[str, Any]) -> str:
    r = result or {}
    level = r.get("exposure", "low")
    inputs = r.get("primary_inputs")
    inputs_cell = inputs if inputs else "none material"
    return (
        "\n\n## Commodity input exposure (from sector/industry classification)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Exposure** | **{level}** |\n"
        f"| Primary inputs | {inputs_cell} |\n"
        f"| Basis | {r.get('rationale', 'n/a')} |\n\n"
        "*Use the exposure level and inputs verbatim. Input-cost sensitivity feeds "
        "gross-margin risk; a 'low' exposure is a valid finding, not a data gap.*\n"
    )
