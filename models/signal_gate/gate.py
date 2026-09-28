"""ML Signal Gate — XGBoost-based trade quality filter.

Trains on historical trade features (strategy, regime, volume, RSI, ADX, VIX,
FII flow) to predict whether a signal will be profitable. Filters out
low-confidence trades before execution.

This operates as a post-strategy, pre-execution gate.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "signal_gate_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


# ─── Feature Engineering ─────────────────────────────────

def extract_features(signal: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, float]:
    """Extract numeric features from a trade signal and trading state.

    Features:
    - confidence: strategy's own confidence score
    - rr_ratio: risk/reward ratio (target - price) / (price - stop_loss)
    - regime_code: HMM regime encoded as int (0-3)
    - vix: India VIX
    - adx: last ADX value
    - rsi: last RSI value
    - volume_ratio: current volume / 20-day average
    - fii_net: FII net buy/sell (crores)
    - strategy_code: strategy encoded as int
    """
    state = state or {}
    symbol = signal.get("symbol", "")
    features: dict[str, float] = {}

    # Signal-level features
    features["confidence"] = float(signal.get("confidence", 50))

    price = float(signal.get("price", 0))
    sl = float(signal.get("stop_loss", 0))
    tgt = float(signal.get("target", 0))

    risk = price - sl if sl > 0 and price > sl else 1.0
    reward = tgt - price if tgt > 0 and tgt > price else 0.0
    features["rr_ratio"] = reward / risk if risk > 0 else 0.0

    # Regime encoding
    regime_map = {
        "LOW_VOL_TREND": 0, "HIGH_VOL_MEAN_REVERT": 1,
        "SYSTEMIC_PANIC": 2, "BULL_EUPHORIA": 3, "UNKNOWN": -1,
    }
    regime = state.get("hmm_regime", {})
    features["regime_code"] = float(regime_map.get(
        regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else "UNKNOWN", -1
    ))

    # VIX
    features["vix"] = float(state.get("india_vix", 0))

    # Market data features for the symbol
    market_data = state.get("market_data", {})
    df = market_data.get(symbol)
    if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 20:
        try:
            import pandas_ta as ta

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            # ADX
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None:
                adx_col = [c for c in adx_df.columns if "ADX" in c and "DM" not in c]
                features["adx"] = float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 0.0
            else:
                features["adx"] = 0.0

            # RSI
            rsi = ta.rsi(close, length=14)
            features["rsi"] = float(rsi.iloc[-1]) if rsi is not None and len(rsi) > 0 else 50.0

            # Volume ratio
            vol_avg = volume.iloc[-20:].mean()
            features["volume_ratio"] = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

        except Exception:
            features.setdefault("adx", 0.0)
            features.setdefault("rsi", 50.0)
            features.setdefault("volume_ratio", 1.0)
    else:
        features["adx"] = 0.0
        features["rsi"] = 50.0
        features["volume_ratio"] = 1.0

    # FII net flow
    fii_dii = state.get("fii_dii_data", {})
    features["fii_net"] = float(fii_dii.get("fii_net", fii_dii.get("fii_net_buy", 0))) if isinstance(fii_dii, dict) else 0.0

    # Strategy encoding
    strategy_map = {"MomentumBreakout": 0, "SupertrendADX": 1, "NiftyOptionsSeller": 2, "MeanReversion": 3}
    features["strategy_code"] = float(strategy_map.get(signal.get("strategy", ""), -1))

    return features


# ─── Training ─────────────────────────────────────────────

def train_signal_gate(
    trades: list[dict[str, Any]],
    feature_dicts: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Train an XGBoost classifier on historical trade outcomes.

    Args:
        trades: List of trade dicts with at least 'net_pnl' or 'pnl' and signal features.
        feature_dicts: Pre-extracted feature dicts. If None, extracts from trade dicts.

    Returns:
        Training report with accuracy, feature importance, model path.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.error("[SignalGate] xgboost/sklearn not installed. Run: pip install xgboost scikit-learn")
        return {"error": "xgboost not installed"}

    if len(trades) < 30:
        logger.warning("[SignalGate] Need >= 30 trades to train, got {}.", len(trades))
        return {"error": f"Insufficient trades: {len(trades)}"}

    # Build feature matrix
    if feature_dicts:
        feature_names = sorted(feature_dicts[0].keys())
        X = np.array([[fd.get(f, 0.0) for f in feature_names] for fd in feature_dicts])
    else:
        # Extract features from trade dicts directly
        feature_names = [
            "confidence", "rr_ratio", "regime_code", "vix",
            "adx", "rsi", "volume_ratio", "fii_net", "strategy_code",
        ]
        rows = []
        for t in trades:
            row = [float(t.get(f, 0)) for f in feature_names]
            rows.append(row)
        X = np.array(rows)

    # Labels: 1 = profitable, 0 = not
    y = np.array([1 if float(t.get("net_pnl", t.get("pnl", 0))) > 0 else 0 for t in trades])

    if len(set(y)) < 2:
        logger.warning("[SignalGate] All trades same outcome — cannot train.")
        return {"error": "No class variance"}

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(trades) // 10), scoring="accuracy")
    logger.info("[SignalGate] CV accuracy: {:.1f}% ± {:.1f}%", cv_scores.mean() * 100, cv_scores.std() * 100)

    # Train final model on all data
    model.fit(X, y)

    # Save model + feature names
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)

    # Feature importance
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))

    report = {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_trades": len(trades),
        "n_profitable": int(y.sum()),
        "feature_importance": importances,
        "model_path": str(MODEL_PATH),
    }

    logger.info("[SignalGate] Model trained and saved: {}", MODEL_PATH)
    return report


# ─── Prediction / Gating ─────────────────────────────────

class SignalGate:
    """Loads trained model and filters signals."""

    def __init__(self, min_probability: float = 0.55) -> None:
        self.min_probability = min_probability
        self._model = None
        self._feature_names: list[str] = []
        self._loaded = False

    def _load(self) -> bool:
        """Lazy-load the trained model."""
        if self._loaded:
            return self._model is not None

        self._loaded = True
        if not MODEL_PATH.exists():
            logger.debug("[SignalGate] No trained model at {} — gate disabled.", MODEL_PATH)
            return False

        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(FEATURE_NAMES_PATH, "r") as f:
                self._feature_names = json.load(f)
            logger.info("[SignalGate] Model loaded ({} features).", len(self._feature_names))
            return True
        except Exception as exc:
            logger.error("[SignalGate] Failed to load model: {}", exc)
            return False

    def should_execute(self, signal: dict[str, Any], state: dict[str, Any] | None = None) -> tuple[bool, float]:
        """Predict whether a signal is worth executing.

        Returns:
            (should_trade, probability) — probability of profitability.
        """
        if not self._load():
            return True, 0.5  # No model — pass all through

        features = extract_features(signal, state)
        x = np.array([[features.get(f, 0.0) for f in self._feature_names]])

        try:
            proba = self._model.predict_proba(x)[0][1]  # P(profitable)
            should = proba >= self.min_probability
            return should, float(proba)
        except Exception as exc:
            logger.error("[SignalGate] Prediction error: {} — passing through.", exc)
            return True, 0.5

    def filter_signals(
        self, signals: list[dict[str, Any]], state: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Filter a list of signals, keeping only those above the probability threshold."""
        if not self._load():
            return signals

        filtered = []
        for sig in signals:
            should, proba = self.should_execute(sig, state)
            sig["ml_gate_proba"] = round(proba, 3)
            if should:
                filtered.append(sig)
            else:
                logger.debug(
                    "[SignalGate] Filtered out {} {} (proba={:.1%} < {:.0%})",
                    sig.get("action", "?"), sig.get("symbol", "?"),
                    proba, self.min_probability,
                )

        logger.info("[SignalGate] {}/{} signals passed gate.", len(filtered), len(signals))
        return filtered


# Singleton
signal_gate = SignalGate()
