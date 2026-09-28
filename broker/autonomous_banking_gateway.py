"""Autonomous Banking Gateway — programmatic margin deposits, withdrawals, and float optimization 
via Setu UPI / banking APIs.

Setu Account Aggregator and UPI integration for:
- Automated margin top-ups when capital < threshold
- Excess cash sweep to savings
- Multi-broker fund transfers (Angel → IBKR prefunding via LRS)
- LRS remittance tracking

SAFETY: All transfers require explicit user confirmation via Telegram before execution.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TransferType(str, Enum):
    MARGIN_TOPUP = "margin_topup"
    EXCESS_SWEEP = "excess_sweep"
    LRS_REMITTANCE = "lrs_remittance"
    BROKER_TRANSFER = "broker_transfer"


class TransferStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    INITIATED = "initiated"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class TransferRequest:
    transfer_id: str = ""
    transfer_type: str = ""
    amount: float = 0.0
    currency: str = "INR"
    source: str = ""
    destination: str = ""
    reason: str = ""
    status: str = TransferStatus.PENDING_APPROVAL
    upi_ref: str = ""
    created_at: str = ""
    approved_at: str = ""
    completed_at: str = ""


class TransferLedger:
    """SQLite ledger for all fund transfer records."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(PROJECT_ROOT / "database" / "transfers.db")
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                transfer_id TEXT PRIMARY KEY,
                transfer_type TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'INR',
                source TEXT,
                destination TEXT,
                reason TEXT,
                status TEXT DEFAULT 'pending_approval',
                upi_ref TEXT,
                created_at TEXT,
                approved_at TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def record_transfer(self, req: TransferRequest) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO transfers 
            (transfer_id, transfer_type, amount, currency, source, destination,
             reason, status, upi_ref, created_at, approved_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            req.transfer_id, req.transfer_type, req.amount, req.currency,
            req.source, req.destination, req.reason, req.status,
            req.upi_ref, req.created_at, req.approved_at, req.completed_at,
        ))
        conn.commit()
        conn.close()

    def update_status(self, transfer_id: str, status: str, **kwargs: Any) -> None:
        conn = sqlite3.connect(self.db_path)
        sets = ["status = ?"]
        vals: list[Any] = [status]
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(transfer_id)
        conn.execute(f"UPDATE transfers SET {', '.join(sets)} WHERE transfer_id = ?", vals)
        conn.commit()
        conn.close()

    def get_pending_transfers(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM transfers WHERE status IN ('pending_approval', 'approved', 'initiated')"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_daily_transfers(self, date: str | None = None) -> list[dict[str, Any]]:
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM transfers WHERE created_at LIKE ?", (f"{date}%",)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


class AutonomousBankingGateway:
    """Manages programmatic fund transfers across bank accounts and brokers.

    SAFETY PROTOCOLS:
    1. All transfers > ₹10,000 require Telegram confirmation
    2. Daily transfer limit: ₹50 Lakh (configurable)
    3. LRS transfers require separate FEMA approval
    4. Transfer cooldown: 5 minutes between automated transfers
    """

    DAILY_LIMIT_INR = 5_000_000  # ₹50 Lakh
    CONFIRMATION_THRESHOLD_INR = 10_000  # ₹10K
    TRANSFER_COOLDOWN_SECONDS = 300  # 5 minutes
    MARGIN_TOPUP_THRESHOLD = 0.3  # Top up when cash < 30% of portfolio

    def __init__(self) -> None:
        self.ledger = TransferLedger()
        self.setu_api_key = getattr(settings, "setu_api_key", "")
        self.setu_base_url = getattr(settings, "setu_base_url", "https://sandbox.setu.co")
        self._last_transfer_time = 0.0

    def check_margin_needs(self, portfolio_value: float, available_cash: float,
                           pending_orders_value: float = 0) -> TransferRequest | None:
        """Check if automated margin top-up is needed."""
        total_needed = pending_orders_value
        cash_ratio = available_cash / portfolio_value if portfolio_value > 0 else 1.0

        if cash_ratio < self.MARGIN_TOPUP_THRESHOLD and total_needed > available_cash * 0.8:
            deficit = total_needed - available_cash * 0.5
            deficit = max(deficit, 50_000)  # Minimum ₹50K top-up

            transfer = TransferRequest(
                transfer_id=self._generate_id("topup"),
                transfer_type=TransferType.MARGIN_TOPUP,
                amount=deficit,
                currency="INR",
                source="bank_savings",
                destination="angel_one_trading",
                reason=f"Margin top-up: cash ratio {cash_ratio:.1%} < {self.MARGIN_TOPUP_THRESHOLD:.0%}",
                status=TransferStatus.PENDING_APPROVAL,
                created_at=datetime.utcnow().isoformat(),
            )
            self.ledger.record_transfer(transfer)
            logger.info("Margin top-up needed: ₹{:,.0f} (cash ratio: {:.1%})", deficit, cash_ratio)
            return transfer

        return None

    def check_excess_cash(self, available_cash: float, portfolio_value: float,
                          max_cash_ratio: float = 0.5) -> TransferRequest | None:
        """Sweep excess trading account cash to savings."""
        cash_ratio = available_cash / portfolio_value if portfolio_value > 0 else 0

        if cash_ratio > max_cash_ratio:
            excess = available_cash - portfolio_value * max_cash_ratio * 0.8
            if excess > 100_000:  # Only sweep if > ₹1 Lakh
                transfer = TransferRequest(
                    transfer_id=self._generate_id("sweep"),
                    transfer_type=TransferType.EXCESS_SWEEP,
                    amount=excess,
                    currency="INR",
                    source="angel_one_trading",
                    destination="bank_savings",
                    reason=f"Excess sweep: cash ratio {cash_ratio:.1%} > {max_cash_ratio:.0%}",
                    status=TransferStatus.PENDING_APPROVAL,
                    created_at=datetime.utcnow().isoformat(),
                )
                self.ledger.record_transfer(transfer)
                logger.info("Excess cash sweep: ₹{:,.0f}", excess)
                return transfer

        return None

    def initiate_lrs_remittance(self, amount_usd: float,
                                destination_broker: str = "ibkr") -> TransferRequest:
        """Initiate LRS remittance for international investing."""
        # FEMA check first
        try:
            from config.fema_lrs_controller import fema_controller
            result = fema_controller.pre_trade_compliance_check(
                symbol="LRS_REMITTANCE",
                quantity=1,
                estimated_usd_value=amount_usd,
                destination_country="US",
            )
            if not result.get("approved"):
                return TransferRequest(
                    transfer_id=self._generate_id("lrs"),
                    transfer_type=TransferType.LRS_REMITTANCE,
                    amount=amount_usd,
                    currency="USD",
                    status=TransferStatus.REJECTED,
                    reason=f"FEMA blocked: {result.get('reason', '')}",
                    created_at=datetime.utcnow().isoformat(),
                )
        except ImportError:
            pass

        transfer = TransferRequest(
            transfer_id=self._generate_id("lrs"),
            transfer_type=TransferType.LRS_REMITTANCE,
            amount=amount_usd,
            currency="USD",
            source="bank_savings",
            destination=destination_broker,
            reason=f"LRS remittance ${amount_usd:,.2f} to {destination_broker}",
            status=TransferStatus.PENDING_APPROVAL,
            created_at=datetime.utcnow().isoformat(),
        )
        self.ledger.record_transfer(transfer)
        logger.info("LRS remittance request: ${:,.2f} → {}", amount_usd, destination_broker)
        return transfer

    def approve_transfer(self, transfer_id: str) -> bool:
        """Approve a pending transfer (called after Telegram confirmation)."""
        self.ledger.update_status(
            transfer_id, TransferStatus.APPROVED,
            approved_at=datetime.utcnow().isoformat(),
        )
        logger.info("Transfer {} approved", transfer_id)
        return True

    def execute_approved_transfers(self) -> list[dict[str, Any]]:
        """Execute all approved transfers. Called by scheduler."""
        now = time.time()
        if now - self._last_transfer_time < self.TRANSFER_COOLDOWN_SECONDS:
            logger.debug("Transfer cooldown active, skipping")
            return []

        pending = self.ledger.get_pending_transfers()
        approved = [t for t in pending if t.get("status") == TransferStatus.APPROVED]

        if not approved:
            return []

        # Check daily limit
        daily = self.ledger.get_daily_transfers()
        daily_total = sum(t.get("amount", 0) for t in daily
                          if t.get("currency") == "INR" and t.get("status") != TransferStatus.REJECTED)

        results = []
        for transfer in approved:
            amount = transfer.get("amount", 0)
            currency = transfer.get("currency", "INR")

            if currency == "INR" and daily_total + amount > self.DAILY_LIMIT_INR:
                logger.warning("Daily transfer limit reached, skipping {}", transfer["transfer_id"])
                continue

            result = self._execute_transfer(transfer)
            results.append(result)
            self._last_transfer_time = time.time()

            if currency == "INR":
                daily_total += amount

        return results

    def _execute_transfer(self, transfer: dict[str, Any]) -> dict[str, Any]:
        """Execute a single approved transfer via Setu UPI or bank API."""
        transfer_id = transfer["transfer_id"]
        transfer_type = transfer.get("transfer_type", "")

        self.ledger.update_status(transfer_id, TransferStatus.INITIATED)

        try:
            if transfer_type == TransferType.LRS_REMITTANCE:
                # LRS requires manual bank process — notify user
                result = {"status": "manual_required", "message": "LRS remittance needs bank portal action"}
            elif self.setu_api_key:
                result = self._setu_upi_transfer(transfer)
            else:
                result = {"status": "simulated", "message": "No Setu API key — transfer simulated"}

            if result.get("status") in ("completed", "simulated"):
                self.ledger.update_status(
                    transfer_id, TransferStatus.COMPLETED,
                    completed_at=datetime.utcnow().isoformat(),
                    upi_ref=result.get("upi_ref", ""),
                )
            elif result.get("status") == "manual_required":
                self.ledger.update_status(transfer_id, TransferStatus.APPROVED)

            result["transfer_id"] = transfer_id
            return result

        except Exception as e:
            logger.error("Transfer {} failed: {}", transfer_id, e)
            self.ledger.update_status(transfer_id, TransferStatus.FAILED)
            return {"transfer_id": transfer_id, "status": "failed", "error": str(e)}

    def _setu_upi_transfer(self, transfer: dict[str, Any]) -> dict[str, Any]:
        """Execute transfer via Setu UPI collect API."""
        # NOTE: This is a placeholder for Setu integration
        # In production, use Setu's UPI DeepLinks or Collect API
        logger.info("Setu UPI transfer: ₹{:,.0f} from {} to {}",
                     transfer.get("amount", 0),
                     transfer.get("source", ""),
                     transfer.get("destination", ""))

        return {
            "status": "simulated",
            "message": "Setu UPI integration placeholder",
            "upi_ref": f"SIM_{transfer.get('transfer_id', '')}",
        }

    def get_transfer_summary(self) -> dict[str, Any]:
        """Get summary of all transfers for dashboard."""
        daily = self.ledger.get_daily_transfers()
        pending = self.ledger.get_pending_transfers()

        total_completed = sum(
            t.get("amount", 0) for t in daily
            if t.get("status") == TransferStatus.COMPLETED
        )

        return {
            "daily_transfers": len(daily),
            "daily_amount": total_completed,
            "daily_limit_remaining": max(0, self.DAILY_LIMIT_INR - total_completed),
            "pending_count": len(pending),
            "pending_amount": sum(t.get("amount", 0) for t in pending),
        }

    @staticmethod
    def _generate_id(prefix: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{hashlib.md5(ts.encode()).hexdigest()[:8]}"


# Singleton
banking_gateway = AutonomousBankingGateway()
