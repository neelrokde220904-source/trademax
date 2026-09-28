"""Evolutionary Alpha Miner — autonomous quant researcher using Agentic Reinforcement Learning (ARL).

Continuously discovers proprietary mathematical relationships (alpha factors) that competitors don't have.
Runs OVERNIGHT ONLY (8 PM — 8 AM IST, never during market hours).

Pipeline: [Idea Agent] → [Factor Agent] → [Eval Agent] → [Reflection Agent] → [Alpha Library]
                ↑                                                    |
                └──────────── Mutation Feedback ─────────────────────┘
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Safe operator library — prevents the LLM from inventing invalid functions
OPERATOR_LIBRARY = """
Available pandas-ta functions: ema(), rsi(), macd(), bbands(), atr(), vwap(), 
                               supertrend(), adx(), obv(), stoch()
Available numpy operations: np.log(), np.diff(), np.percentile(), np.corrcoef()
Available cross-sectional: rank(), zscore(), winsorize()
Available time-series: rolling(n).mean(), rolling(n).std(), shift(n), pct_change(n)
Available group operations: groupby(sector).transform()
"""


def _call_claude(model: str, prompt: str, max_tokens: int = 2000) -> str:
    if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key:
        return ""
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


class IdeaAgent:
    """Generate novel, testable trading hypotheses using LLM + optional RAG over research papers."""

    def generate_hypothesis(self, context: str = "") -> dict[str, Any]:
        prompt = f"""Based on recent quantitative finance research:
{context[:2000] if context else 'Use your knowledge of quant finance.'}

Generate ONE novel, testable trading hypothesis for NSE Indian equities.
The hypothesis must:
1. Be mathematically specific (not "buy good stocks")
2. Be non-obvious (not RSI, MACD, or standard TA)
3. Have a logical economic rationale
4. Be implementable in Python with pandas-ta and OHLCV data

Examples of good hypotheses:
- "Order flow imbalance in the last 15 min predicts next-day gap direction"
- "Stocks where promoter buying coincides with FII selling show reversal within 5 days"
- "NSE options PCR divergence from historical mean reverts within 3 sessions"
- "Stocks with volume > 3x average on inside bars show breakout within 2 sessions"
- "5-day momentum factor ranked by sector relative strength predicts 10-day returns"

Output ONLY JSON:
{{"hypothesis": "...", "rationale": "...", "expected_IC": 0.05, "lookback_days": 20}}"""

        response = _call_claude("claude-haiku-4-5-20251001", prompt, max_tokens=500)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {"hypothesis": "Failed to generate", "rationale": "", "expected_IC": 0.0}


class FactorAgent:
    """Translate natural language hypotheses into executable Python code."""

    def generate_factor_code(self, hypothesis: str) -> str:
        prompt = f"""Translate this trading hypothesis into executable Python code:
Hypothesis: {hypothesis}

Available operators ONLY (do not invent functions):
{OPERATOR_LIBRARY}

Requirements:
- Function signature: def alpha_factor(df: pd.DataFrame) -> pd.Series
- df has columns: open, high, low, close, volume (all lowercase)
- Returns a Series of float signals (positive = bullish, negative = bearish)
- Include input validation (check df is not empty, required columns exist)
- No lookahead bias (never use future data — no .shift(-n))
- Use try/except for robustness
- Import pandas as pd, numpy as np, pandas_ta as ta at the top

Output ONLY the raw Python code, no markdown, no explanation, no ```."""

        code = _call_claude(settings.llm_fast_model, prompt, max_tokens=1500)

        # Strip markdown fences if present
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Validate
        self._validate_syntax(code)
        self._check_lookahead_bias(code)

        return code

    @staticmethod
    def _validate_syntax(code: str) -> None:
        """Validate Python syntax without executing."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}") from e

    @staticmethod
    def _check_lookahead_bias(code: str) -> None:
        """Check for common lookahead bias patterns."""
        dangerous = [".shift(-", "future", "tomorrow", "next_day"]
        code_lower = code.lower()
        for pattern in dangerous:
            if pattern in code_lower:
                raise ValueError(f"Potential lookahead bias detected: '{pattern}' found in code")


class EvalAgent:
    """Rigorously backtest generated alpha factors on historical data."""

    # Promotion criteria
    MIN_IC = 0.03
    MIN_SHARPE = 1.5
    MIN_WIN_RATE = 0.55
    MAX_DRAWDOWN = 0.15

    def evaluate_factor(self, factor_code: str, price_data: dict[str, pd.DataFrame],
                        universe: list[str] | None = None) -> dict[str, Any]:
        """Backtest a factor on historical data and return metrics."""
        results: dict[str, Any] = {"signals": {}, "returns": []}

        symbols = universe or list(price_data.keys())[:10]

        for symbol in symbols:
            df = price_data.get(symbol)
            if df is None or len(df) < 60:
                continue

            try:
                signals = self._safe_execute(factor_code, df)
                if signals is not None and len(signals) > 0:
                    results["signals"][symbol] = signals
            except Exception as e:
                logger.debug("Factor eval failed on {}: {}", symbol, e)

        if not results["signals"]:
            return {
                "IC": 0.0, "sharpe": 0.0, "max_drawdown": 1.0,
                "win_rate": 0.0, "promoted": False,
                "failure_analysis": "Factor produced no valid signals",
            }

        metrics = self._compute_metrics(results["signals"], price_data)
        metrics["promoted"] = (
            metrics["IC"] > self.MIN_IC
            and metrics["sharpe"] > self.MIN_SHARPE
            and metrics["win_rate"] > self.MIN_WIN_RATE
            and metrics["max_drawdown"] < self.MAX_DRAWDOWN
        )

        return metrics

    def _safe_execute(self, code: str, df: pd.DataFrame) -> pd.Series | None:
        """Execute factor code in isolated subprocess with timeout."""
        # Write code to temp file and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            wrapper = f"""
import pandas as pd
import numpy as np
import sys
import json

try:
    import pandas_ta as ta
except ImportError:
    ta = None

{code}

# Read data
df = pd.read_json(sys.stdin.read())
result = alpha_factor(df)
if result is not None:
    print(json.dumps(result.fillna(0).tolist()))
else:
    print("null")
"""
            f.write(wrapper)
            f.flush()
            temp_path = f.name

        try:
            df_json = df.to_json()
            proc = subprocess.run(
                [sys.executable, temp_path],
                input=df_json, capture_output=True, text=True,
                timeout=30,  # 30 second timeout
            )
            if proc.returncode == 0 and proc.stdout.strip() != "null":
                values = json.loads(proc.stdout)
                return pd.Series(values, dtype=float)
        except subprocess.TimeoutExpired:
            logger.warning("Factor execution timed out")
        except Exception as e:
            logger.debug("Factor execution error: {}", e)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return None

    def _compute_metrics(self, signals: dict[str, pd.Series],
                         price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Compute alpha factor quality metrics."""
        all_ic = []
        all_returns = []

        for symbol, signal_series in signals.items():
            df = price_data.get(symbol)
            if df is None or len(df) < len(signal_series):
                continue

            # Forward returns (1-day)
            close = df["close"].values[-len(signal_series):]
            fwd_returns = np.diff(close) / close[:-1]

            # Align
            sig = np.array(signal_series)[:-1] if len(signal_series) > len(fwd_returns) else np.array(signal_series)
            fwd = fwd_returns[:len(sig)]

            if len(sig) > 10 and len(fwd) > 10:
                # IC = rank correlation between signal and forward returns
                ic = np.corrcoef(sig, fwd)[0, 1] if np.std(sig) > 0 else 0
                all_ic.append(ic)

                # Strategy returns: long positive signal, short negative
                position = np.sign(sig)
                strat_returns = position * fwd
                all_returns.extend(strat_returns.tolist())

        if not all_ic or not all_returns:
            return {"IC": 0.0, "sharpe": 0.0, "max_drawdown": 1.0, "win_rate": 0.0}

        returns_arr = np.array(all_returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr) if np.std(returns_arr) > 0 else 1e-8

        # Equity curve for drawdown
        equity = np.cumprod(1 + returns_arr)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        wins = np.sum(returns_arr > 0)
        total = len(returns_arr)

        return {
            "IC": float(np.mean(all_ic)),
            "sharpe": float(mean_return / std_return * np.sqrt(252)),
            "max_drawdown": max_dd,
            "win_rate": float(wins / total) if total > 0 else 0.0,
            "total_return": float(np.prod(1 + returns_arr) - 1),
            "n_trades": total,
        }


class ReflectionAgent:
    """Analyze why a factor failed and generate specific mutation instructions."""

    def mutate_factor(self, factor_code: str, eval_results: dict[str, Any]) -> str:
        """Apply ONE specific mutation to fix the primary failure mode."""
        prompt = f"""This trading factor failed evaluation:

Code:
{factor_code[:2000]}

Performance metrics:
- IC: {eval_results.get('IC', 0):.4f} (target > 0.03)
- Sharpe: {eval_results.get('sharpe', 0):.2f} (target > 1.5)
- Max Drawdown: {eval_results.get('max_drawdown', 0):.2%} (limit < 15%)
- Win Rate: {eval_results.get('win_rate', 0):.1%} (target > 55%)

Specific failure: {eval_results.get('failure_analysis', 'See metrics')}

Apply ONE specific mutation to fix the primary failure:
- If IC is low: add a volatility filter (ATR) or change lookback period
- If drawdown is high: add a regime filter (only active when volatility < threshold)
- If win rate is low: add confirmation signal (volume > 2x average)
- If sharpe is low: reduce signal frequency with stricter entry conditions

Output ONLY the mutated Python code, no markdown, no explanation."""

        code = _call_claude(settings.llm_fast_model, prompt, max_tokens=1500)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return code


class AlphaLibrary:
    """Store, rank, and manage the evolving collection of trading strategies."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(PROJECT_ROOT / "alpha_library" / "alpha_store.db")
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alpha_factors (
                id TEXT PRIMARY KEY,
                hypothesis TEXT,
                code TEXT NOT NULL,
                ic REAL DEFAULT 0,
                sharpe REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_return REAL DEFAULT 0,
                generation INTEGER DEFAULT 0,
                parent_id TEXT,
                created_at TEXT,
                last_evaluated TEXT,
                is_active BOOLEAN DEFAULT 1,
                rolling_ic REAL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def promote_factor(self, factor_code: str, metrics: dict[str, Any],
                       hypothesis: str = "", generation: int = 0,
                       parent_id: str = "") -> str:
        """Add a successful factor to the library."""
        factor_id = hashlib.md5(factor_code.encode()).hexdigest()[:12]
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO alpha_factors 
            (id, hypothesis, code, ic, sharpe, max_drawdown, win_rate, total_return,
             generation, parent_id, created_at, last_evaluated, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            factor_id, hypothesis, factor_code,
            metrics.get("IC", 0), metrics.get("sharpe", 0),
            metrics.get("max_drawdown", 0), metrics.get("win_rate", 0),
            metrics.get("total_return", 0),
            generation, parent_id,
            datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
        logger.info("Promoted alpha factor {} (IC={:.4f}, Sharpe={:.2f})",
                     factor_id, metrics.get("IC", 0), metrics.get("sharpe", 0))
        return factor_id

    def deprecate_decayed_factors(self, min_rolling_ic: float = 0.01) -> int:
        """Retire factors whose rolling IC has decayed below threshold."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "UPDATE alpha_factors SET is_active = 0 WHERE is_active = 1 AND rolling_ic < ?",
            (min_rolling_ic,),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        if count > 0:
            logger.info("Deprecated {} decayed alpha factors (rolling_ic < {})", count, min_rolling_ic)
        return count

    def get_active_factors(self) -> list[dict[str, Any]]:
        """Return currently active factors for live trading integration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alpha_factors WHERE is_active = 1 ORDER BY sharpe DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_factor_count(self) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        active = conn.execute("SELECT COUNT(*) FROM alpha_factors WHERE is_active = 1").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM alpha_factors").fetchone()[0]
        conn.close()
        return {"active": active, "total": total}


class EvolutionaryAlphaMiner:
    """Complete ARL pipeline: Idea → Factor → Eval → Reflect → Mutate → Library.

    CONSTRAINT: Runs OVERNIGHT ONLY (8 PM — 8 AM IST).
    Never run during market hours.
    """

    MAX_GENERATIONS = 5  # Maximum mutation iterations per hypothesis
    MAX_HYPOTHESES_PER_RUN = 3  # Hypotheses per overnight run

    def __init__(self) -> None:
        self.idea_agent = IdeaAgent()
        self.factor_agent = FactorAgent()
        self.eval_agent = EvalAgent()
        self.reflection_agent = ReflectionAgent()
        self.alpha_library = AlphaLibrary()

    def run_overnight_cycle(self, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Run the complete ARL loop. Call this from the overnight scheduler."""
        results: dict[str, Any] = {
            "hypotheses_tested": 0,
            "factors_promoted": 0,
            "factors_failed": 0,
            "details": [],
        }

        for i in range(self.MAX_HYPOTHESES_PER_RUN):
            logger.info("Alpha miner: hypothesis {}/{}", i + 1, self.MAX_HYPOTHESES_PER_RUN)

            try:
                detail = self._evolve_single_hypothesis(price_data)
                results["details"].append(detail)
                results["hypotheses_tested"] += 1

                if detail.get("promoted"):
                    results["factors_promoted"] += 1
                else:
                    results["factors_failed"] += 1

            except Exception as e:
                logger.error("Alpha miner hypothesis {} failed: {}", i + 1, e)
                results["factors_failed"] += 1

        # Weekly maintenance: deprecate decayed factors
        deprecated = self.alpha_library.deprecate_decayed_factors()
        results["deprecated"] = deprecated

        logger.info(
            "Alpha miner overnight complete: {} tested, {} promoted, {} failed, {} deprecated",
            results["hypotheses_tested"], results["factors_promoted"],
            results["factors_failed"], deprecated,
        )
        return results

    def _evolve_single_hypothesis(self, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Generate, test, and evolve a single hypothesis."""
        # Step 1: Generate hypothesis
        hypothesis = self.idea_agent.generate_hypothesis()
        hyp_text = hypothesis.get("hypothesis", "")
        logger.info("Hypothesis: {}", hyp_text[:100])

        # Step 2: Generate factor code
        try:
            factor_code = self.factor_agent.generate_factor_code(hyp_text)
        except ValueError as e:
            return {"hypothesis": hyp_text, "error": str(e), "promoted": False}

        # Step 3: Evaluate
        metrics = self.eval_agent.evaluate_factor(factor_code, price_data)

        # Step 4: If promoted, store in library
        if metrics.get("promoted"):
            factor_id = self.alpha_library.promote_factor(factor_code, metrics, hyp_text)
            return {
                "hypothesis": hyp_text, "metrics": metrics,
                "promoted": True, "factor_id": factor_id, "generation": 0,
            }

        # Step 5: Mutation loop
        current_code = factor_code
        for gen in range(1, self.MAX_GENERATIONS + 1):
            logger.info("Mutation generation {}/{}", gen, self.MAX_GENERATIONS)

            try:
                mutated_code = self.reflection_agent.mutate_factor(current_code, metrics)
                FactorAgent._validate_syntax(mutated_code)
                FactorAgent._check_lookahead_bias(mutated_code)
            except ValueError as e:
                logger.debug("Mutation {} invalid: {}", gen, e)
                continue

            metrics = self.eval_agent.evaluate_factor(mutated_code, price_data)

            if metrics.get("promoted"):
                factor_id = self.alpha_library.promote_factor(
                    mutated_code, metrics, hyp_text, generation=gen,
                )
                return {
                    "hypothesis": hyp_text, "metrics": metrics,
                    "promoted": True, "factor_id": factor_id, "generation": gen,
                }

            current_code = mutated_code

        return {"hypothesis": hyp_text, "metrics": metrics, "promoted": False}


# Singleton
alpha_miner = EvolutionaryAlphaMiner()
