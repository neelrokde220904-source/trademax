"""FEMA / LRS Compliance Engine — legally mandatory for Indian residents trading foreign markets.

FEMA = Foreign Exchange Management Act
LRS = Liberalised Remittance Scheme (RBI)

HARD RULES (violations carry criminal liability):
1. Annual limit: $250,000 USD per financial year per individual
2. PROHIBITED: Any foreign derivatives (options, futures, margins) — ABSOLUTE BAN
3. TCS: 20% tax collected at source on remittances > ₹10 lakh for investment
4. FATF: No remittances to FATF blacklisted/grey-listed countries
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LRSLedger:
    """Persistent LRS remittance tracking using SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS remittances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                amount_inr REAL NOT NULL,
                exchange_rate REAL NOT NULL,
                purpose TEXT NOT NULL,
                broker TEXT NOT NULL,
                tcs_applicable BOOLEAN DEFAULT 0,
                tcs_amount_inr REAL DEFAULT 0.0,
                financial_year TEXT NOT NULL,
                status TEXT DEFAULT 'completed'
            )
        """)
        conn.commit()
        conn.close()

    def record(self, amount_usd: float, purpose: str, broker: str,
               amount_inr: float = 0.0, exchange_rate: float = 0.0,
               tcs_applicable: bool = False, tcs_amount_inr: float = 0.0,
               timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.utcnow()
        fy = self._get_financial_year(ts)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO remittances 
            (timestamp, amount_usd, amount_inr, exchange_rate, purpose, broker,
             tcs_applicable, tcs_amount_inr, financial_year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts.isoformat(), amount_usd, amount_inr, exchange_rate,
              purpose, broker, tcs_applicable, tcs_amount_inr, fy))
        conn.commit()
        conn.close()

    def get_fy_total_usd(self, financial_year: str | None = None) -> float:
        fy = financial_year or self._get_financial_year(datetime.utcnow())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COALESCE(SUM(amount_usd), 0) FROM remittances WHERE financial_year = ?",
            (fy,)
        )
        total = cursor.fetchone()[0]
        conn.close()
        return total

    def get_fy_tcs_total(self, financial_year: str | None = None) -> float:
        fy = financial_year or self._get_financial_year(datetime.utcnow())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COALESCE(SUM(tcs_amount_inr), 0) FROM remittances WHERE financial_year = ?",
            (fy,)
        )
        total = cursor.fetchone()[0]
        conn.close()
        return total

    @staticmethod
    def _get_financial_year(dt: datetime) -> str:
        if dt.month >= 4:
            return f"FY{dt.year}-{dt.year + 1}"
        return f"FY{dt.year - 1}-{dt.year}"


class FEMALRSController:
    """FEMA/LRS compliance controller.

    Hard-coded gate — no override possible. This ensures legal compliance
    for Indian residents investing in foreign markets.
    """

    # ABSOLUTELY PROHIBITED instrument types under FEMA
    PROHIBITED_INSTRUMENTS = frozenset({
        "OPTIONS", "FUTURES", "MARGIN", "CFD", "DERIVATIVES",
        "FOREX_MARGIN", "SPREAD_BET", "BINARY_OPTIONS",
    })

    # FATF blacklisted / grey-listed countries (update periodically)
    FATF_RESTRICTED: set[str] = set()

    def __init__(self, db_path: str | None = None) -> None:
        _db = db_path or str(PROJECT_ROOT / "compliance" / "lrs_ledger.db")
        self.ledger = LRSLedger(_db)
        self.annual_limit_usd = getattr(settings, "lrs_annual_limit_usd", 250_000)
        self.tcs_threshold_inr = getattr(settings, "tcs_threshold_inr", 10_00_000)
        self.FINANCIAL_YEAR_START_MONTH = 4  # April 1 in India
        self._load_fatf_watchlist()

    def _load_fatf_watchlist(self) -> None:
        fatf_path = PROJECT_ROOT / "compliance" / "fatf_watchlist.json"
        if fatf_path.exists():
            with open(fatf_path) as f:
                data = json.load(f)
                self.FATF_RESTRICTED = set(data.get("blacklisted", []) + data.get("grey_listed", []))
        else:
            # Default restricted set
            self.FATF_RESTRICTED = {
                "DPRK", "IRAN", "MYANMAR",  # FATF blacklist
                "SYRIA", "YEMEN", "LIBYA",  # High-risk
            }

    def check_compliance(self, proposed_remittance_usd: float,
                         instrument_type: str,
                         destination_country: str = "US",
                         exchange_rate_inr_usd: float = 83.5) -> dict[str, Any]:
        """Check if a proposed international trade/remittance is FEMA/LRS compliant.

        Returns:
            {
                "approved": bool,
                "reason": str,
                "lrs_used_usd": float,
                "lrs_remaining_usd": float,
                "tcs_applicable": bool,
                "tcs_amount_inr": float,
                "warnings": list[str]
            }
        """
        warnings: list[str] = []

        # CHECK 1: Foreign derivatives are ABSOLUTELY BANNED
        if instrument_type.upper() in self.PROHIBITED_INSTRUMENTS:
            logger.error(
                "FEMA VIOLATION BLOCKED: {} trading is prohibited for Indian residents",
                instrument_type,
            )
            return {
                "approved": False,
                "reason": f"FEMA ABSOLUTE BAN: {instrument_type} trading is illegal for Indian residents. "
                          "Only delivery-based equity/ETF buying is permitted on foreign exchanges.",
                "lrs_used_usd": self.ledger.get_fy_total_usd(),
                "lrs_remaining_usd": self.annual_limit_usd - self.ledger.get_fy_total_usd(),
                "tcs_applicable": False,
                "tcs_amount_inr": 0.0,
                "warnings": [],
            }

        # CHECK 2: FATF restricted countries
        if destination_country.upper() in self.FATF_RESTRICTED:
            logger.error("FATF BLOCKED: {} is a restricted jurisdiction", destination_country)
            return {
                "approved": False,
                "reason": f"FATF restricted jurisdiction: {destination_country}. "
                          "Remittances to this country are prohibited.",
                "lrs_used_usd": self.ledger.get_fy_total_usd(),
                "lrs_remaining_usd": self.annual_limit_usd - self.ledger.get_fy_total_usd(),
                "tcs_applicable": False,
                "tcs_amount_inr": 0.0,
                "warnings": [],
            }

        # CHECK 3: LRS annual limit ($250,000)
        fy_used = self.ledger.get_fy_total_usd()
        fy_remaining = self.annual_limit_usd - fy_used

        if proposed_remittance_usd > fy_remaining:
            logger.error(
                "LRS LIMIT EXCEEDED: Proposed ${:,.2f} exceeds remaining ${:,.2f}",
                proposed_remittance_usd, fy_remaining,
            )
            return {
                "approved": False,
                "reason": f"LRS annual limit exceeded. Used: ${fy_used:,.2f}, "
                          f"Remaining: ${fy_remaining:,.2f}, Requested: ${proposed_remittance_usd:,.2f}",
                "lrs_used_usd": fy_used,
                "lrs_remaining_usd": fy_remaining,
                "tcs_applicable": False,
                "tcs_amount_inr": 0.0,
                "warnings": [],
            }

        # CHECK 4: TCS calculation (20% on remittances > ₹10 lakh for investment)
        proposed_inr = proposed_remittance_usd * exchange_rate_inr_usd
        total_fy_inr = (fy_used * exchange_rate_inr_usd) + proposed_inr
        tcs_applicable = total_fy_inr > self.tcs_threshold_inr
        tcs_amount = 0.0

        if tcs_applicable:
            # TCS is 20% on the amount EXCEEDING ₹10 lakh threshold
            taxable_amount = total_fy_inr - self.tcs_threshold_inr
            if taxable_amount > 0:
                tcs_amount = taxable_amount * 0.20
                warnings.append(
                    f"TCS of ₹{tcs_amount:,.2f} (20%) applies on amount exceeding "
                    f"₹{self.tcs_threshold_inr:,.0f} threshold."
                )

        # Proximity warnings
        utilization_pct = ((fy_used + proposed_remittance_usd) / self.annual_limit_usd) * 100
        if utilization_pct > 80:
            warnings.append(
                f"LRS utilization at {utilization_pct:.1f}% after this transaction. "
                f"Only ${fy_remaining - proposed_remittance_usd:,.2f} will remain."
            )

        logger.info(
            "FEMA/LRS compliance check PASSED: ${:,.2f} remittance approved "
            "(FY used: ${:,.2f}, remaining: ${:,.2f})",
            proposed_remittance_usd, fy_used, fy_remaining - proposed_remittance_usd,
        )

        return {
            "approved": True,
            "reason": "Compliant with FEMA/LRS regulations",
            "lrs_used_usd": fy_used,
            "lrs_remaining_usd": fy_remaining - proposed_remittance_usd,
            "tcs_applicable": tcs_applicable,
            "tcs_amount_inr": tcs_amount,
            "warnings": warnings,
        }

    def record_remittance(self, amount_usd: float, purpose: str, broker: str,
                          exchange_rate: float = 83.5) -> None:
        """Called AFTER successful international transfer — updates the ledger."""
        amount_inr = amount_usd * exchange_rate

        # Calculate TCS
        fy_used = self.ledger.get_fy_total_usd()
        total_inr = (fy_used * exchange_rate) + amount_inr
        tcs_applicable = total_inr > self.tcs_threshold_inr
        tcs_amount = max(0, (total_inr - self.tcs_threshold_inr) * 0.20) if tcs_applicable else 0.0

        self.ledger.record(
            amount_usd=amount_usd,
            purpose=purpose,
            broker=broker,
            amount_inr=amount_inr,
            exchange_rate=exchange_rate,
            tcs_applicable=tcs_applicable,
            tcs_amount_inr=tcs_amount,
        )

        logger.info(
            "Remittance recorded: ${:,.2f} to {} via {} (TCS: ₹{:,.2f})",
            amount_usd, purpose, broker, tcs_amount,
        )

    def get_lrs_status(self) -> dict[str, Any]:
        """Get current LRS utilization for dashboard display."""
        fy_used = self.ledger.get_fy_total_usd()
        tcs_accumulated = self.ledger.get_fy_tcs_total()
        return {
            "used_usd": fy_used,
            "remaining_usd": self.annual_limit_usd - fy_used,
            "annual_limit_usd": self.annual_limit_usd,
            "utilization_pct": (fy_used / self.annual_limit_usd) * 100 if self.annual_limit_usd else 0,
            "tcs_accumulated_inr": tcs_accumulated,
            "financial_year": LRSLedger._get_financial_year(datetime.utcnow()),
        }


# Singleton
fema_controller = FEMALRSController()
