"""Hidden Markov Model Regime Classifier — detects market regimes from multi-dimensional features.

The HMM is the GLOBAL MASTER SWITCH for the entire trading system.
Strategies optimized for bull markets cause catastrophic drawdowns in bear markets.

Regimes:
  0: LOW_VOL_TREND      — Deploy momentum + breakout strategies
  1: HIGH_VOL_MEAN_REVERT — Deploy bollinger reversion, range strategies
  2: SYSTEMIC_PANIC       — ALL CASH + hedge only
  3: BULL_EUPHORIA        — Momentum but tighten stops (potential top)

Required: hmmlearn, scikit-learn, numpy, scipy
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. HMM regime classifier will use fallback mode.")


class HMMRegimeClassifier:
    """4-state Gaussian HMM for market regime classification.

    Runs on DAILY data — recompute every morning at 8:30 AM IST.
    """

    REGIMES = {
        0: "LOW_VOL_TREND",
        1: "HIGH_VOL_MEAN_REVERT",
        2: "SYSTEMIC_PANIC",
        3: "BULL_EUPHORIA",
    }

    def __init__(self, n_components: int = 4, model_path: str | None = None) -> None:
        self.n_components = n_components
        self.model_path = Path(model_path or getattr(
            settings, "hmm_model_path",
            str(PROJECT_ROOT / "models" / "hmm_regime_classifier.pkl"),
        ))
        self.is_fitted = False
        self.model = None

        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Load a previously trained model from disk if it exists."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["model"]
                self._regime_map = saved.get("regime_map", {0: 0, 1: 1, 2: 2, 3: 3})
                self.is_fitted = True
                logger.info("Loaded HMM model from {}", self.model_path)
            except Exception as e:
                logger.warning("Failed to load HMM model: {}", e)

    def build_feature_matrix(self, market_data: dict[str, Any]) -> np.ndarray:
        """Build multi-dimensional feature vector for regime classification.

        Args:
            market_data: Dictionary containing market DataFrames and scalar values.
                Expected keys:
                - 'nifty_df': DataFrame with OHLCV for Nifty 50
                - 'india_vix' or 'vix_series': VIX values
                - 'fii_net_series': FII net buy/sell series
                - 'usd_inr_series': USD/INR exchange rate series

        Returns:
            Feature matrix of shape (n_days, n_features).
        """
        nifty_df = market_data.get("nifty_df")
        if nifty_df is None or len(nifty_df) < 50:
            raise ValueError("Need at least 50 days of Nifty data for HMM features")

        close = nifty_df["close"].values.astype(float)

        # Feature 1: Realized volatility (20-day rolling std of log returns)
        log_returns = np.diff(np.log(close))
        realized_vol = np.array([
            np.std(log_returns[max(0, i - 19):i + 1]) * np.sqrt(252)
            for i in range(len(log_returns))
        ])

        # Feature 2: Momentum (20-day return)
        momentum_20d = np.array([
            (close[i + 1] / close[max(0, i - 18)]) - 1.0
            for i in range(len(log_returns))
        ])

        # Feature 3: VIX level (or proxy via realized vol)
        vix_series = market_data.get("vix_series")
        if vix_series is not None and len(vix_series) >= len(log_returns):
            vix_feature = np.array(vix_series[-len(log_returns):], dtype=float)
        else:
            vix_feature = realized_vol * 100  # Proxy

        # Feature 4: Volume change (20-day rolling % change)
        if "volume" in nifty_df.columns:
            volume = nifty_df["volume"].values.astype(float)[1:]  # Align with returns
            vol_ma20 = np.array([
                np.mean(volume[max(0, i - 19):i + 1])
                for i in range(len(volume))
            ])
            volume_change = np.where(vol_ma20 > 0, volume / vol_ma20 - 1.0, 0.0)
        else:
            volume_change = np.zeros(len(log_returns))

        # Feature 5: FII net flow z-score (if available)
        fii_series = market_data.get("fii_net_series")
        if fii_series is not None and len(fii_series) >= len(log_returns):
            fii = np.array(fii_series[-len(log_returns):], dtype=float)
            fii_mean = np.mean(fii) if len(fii) > 0 else 0
            fii_std = np.std(fii) if len(fii) > 0 else 1
            fii_zscore = (fii - fii_mean) / max(fii_std, 1e-8)
        else:
            fii_zscore = np.zeros(len(log_returns))

        # Feature 6: USD/INR % change (currency stress)
        usd_inr = market_data.get("usd_inr_series")
        if usd_inr is not None and len(usd_inr) >= len(log_returns):
            usd_inr_arr = np.array(usd_inr[-len(log_returns):], dtype=float)
            usd_inr_pct = np.diff(usd_inr_arr) / usd_inr_arr[:-1]
            # Pad to match length
            usd_inr_pct = np.concatenate([[0.0], usd_inr_pct])
        else:
            usd_inr_pct = np.zeros(len(log_returns))

        features = np.column_stack([
            realized_vol,
            momentum_20d,
            vix_feature,
            volume_change,
            fii_zscore,
            usd_inr_pct,
        ])

        # Remove NaN rows
        valid_mask = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
        return features[valid_mask]

    def fit(self, historical_features: np.ndarray) -> None:
        """Train HMM on 3+ years of daily data. Run once, then update weekly.

        Args:
            historical_features: Feature matrix from build_feature_matrix().
        """
        if not HMM_AVAILABLE:
            logger.error("hmmlearn not installed. Cannot train HMM.")
            return

        if len(historical_features) < 252:
            logger.warning("HMM training requires >= 252 days. Got {}. Training anyway.", len(historical_features))

        logger.info("Training HMM on {} days of data...", len(historical_features))
        self.model.fit(historical_features)
        self.is_fitted = True

        # Calibrate regime labels based on learned parameters
        self._regime_map = self._calibrate_regime_labels(historical_features)

        # Save model
        self._save_model()
        logger.info("HMM trained and saved. Regime mapping: {}", self._regime_map)

    def _calibrate_regime_labels(self, features: np.ndarray) -> dict[int, int]:
        """Map HMM hidden states to semantic regime labels based on mean volatility.

        States are sorted by mean realized volatility (feature 0):
        - Lowest vol → LOW_VOL_TREND (0)
        - Medium vol → HIGH_VOL_MEAN_REVERT (1) or BULL_EUPHORIA (3)
        - Highest vol → SYSTEMIC_PANIC (2)
        """
        states = self.model.predict(features)
        state_vols = {}
        for state_id in range(self.n_components):
            mask = states == state_id
            if mask.any():
                state_vols[state_id] = np.mean(features[mask, 0])  # Mean realized vol
            else:
                state_vols[state_id] = 0.0

        # Sort by volatility
        sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols[s])

        regime_map = {}
        regime_map[sorted_states[0]] = 0  # LOW_VOL_TREND
        regime_map[sorted_states[-1]] = 2  # SYSTEMIC_PANIC

        # Middle states: use momentum to distinguish
        for s in sorted_states[1:-1]:
            mask = states == s
            if mask.any():
                mean_momentum = np.mean(features[mask, 1])  # Feature 1 = momentum
                if mean_momentum > 0:
                    regime_map[s] = 3  # BULL_EUPHORIA
                else:
                    regime_map[s] = 1  # HIGH_VOL_MEAN_REVERT
            else:
                regime_map[s] = 1

        return regime_map

    def _save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "regime_map": self._regime_map}, f)

    def predict_current_regime(self, features: np.ndarray) -> dict[str, Any]:
        """Predict current market regime from feature matrix.

        Called every morning at 9:00 AM IST before market open.

        Returns:
            {
                "regime": str,           # e.g. "LOW_VOL_TREND"
                "regime_id": int,        # 0-3
                "panic_risk": float,     # Probability of transitioning to SYSTEMIC_PANIC
                "transition_probs": dict, # Probabilities of each regime tomorrow
                "action_directive": dict, # System-wide behavior changes
            }
        """
        if not self.is_fitted:
            logger.warning("HMM not fitted. Returning fallback regime (LOW_VOL_TREND).")
            return self._fallback_regime()

        if not HMM_AVAILABLE:
            return self._fallback_regime()

        raw_state = self.model.predict(features)[-1]
        regime_map = getattr(self, "_regime_map", {i: i for i in range(4)})
        mapped_regime = regime_map.get(raw_state, 0)

        # Transition matrix: probability of moving to each regime tomorrow
        transition_probs = self.model.transmat_[raw_state]

        # Find which raw state maps to SYSTEMIC_PANIC
        panic_raw_states = [k for k, v in regime_map.items() if v == 2]
        panic_probability = sum(transition_probs[s] for s in panic_raw_states) if panic_raw_states else 0.0

        regime_name = self.REGIMES.get(mapped_regime, "UNKNOWN")

        result = {
            "regime": regime_name,
            "regime_id": mapped_regime,
            "panic_risk": float(panic_probability),
            "transition_probs": {
                self.REGIMES.get(regime_map.get(i, i), "UNKNOWN"): float(transition_probs[i])
                for i in range(len(transition_probs))
            },
            "action_directive": self._get_action_directive(mapped_regime, panic_probability),
        }

        logger.info(
            "HMM Regime: {} (panic_risk={:.2%})",
            regime_name, panic_probability,
        )
        return result

    def _get_action_directive(self, regime_id: int, panic_risk: float) -> dict[str, Any]:
        """Translate regime into concrete system-wide behavior changes.

        This dict is passed to every agent in TradingState.
        """
        directives: dict[int, dict[str, Any]] = {
            0: {  # LOW_VOL_TREND
                "allowed_strategies": ["momentum_breakout", "supertrend_adx", "trend_following"],
                "max_position_size_pct": 0.12,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 2.5,
                "allow_new_positions": True,
                "hedge_required": False,
                "max_portfolio_beta": 1.3,
                "description": "Trending market — deploy momentum strategies with standard sizing",
            },
            1: {  # HIGH_VOL_MEAN_REVERT
                "allowed_strategies": ["mean_reversion", "bollinger_reversion", "range_bound"],
                "max_position_size_pct": 0.08,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 1.5,
                "allow_new_positions": True,
                "hedge_required": False,
                "max_portfolio_beta": 1.0,
                "description": "Volatile mean-reverting market — use reversion strategies with tighter sizing",
            },
            2: {  # SYSTEMIC_PANIC
                "allowed_strategies": [],
                "max_position_size_pct": 0.0,
                "stop_loss_multiplier": 0.0,
                "take_profit_multiplier": 0.0,
                "allow_new_positions": False,
                "hedge_required": True,
                "max_portfolio_beta": 0.0,
                "description": "SYSTEMIC PANIC — NO new positions. Hedge existing. Move to cash.",
            },
            3: {  # BULL_EUPHORIA
                "allowed_strategies": ["momentum_breakout", "supertrend_adx"],
                "max_position_size_pct": 0.08,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 2.0,
                "allow_new_positions": True,
                "hedge_required": True,
                "max_portfolio_beta": 0.8,
                "description": "Euphoric market — ride momentum but tighten stops (potential top signal)",
            },
        }

        result = directives.get(regime_id, directives[0]).copy()

        # Override if panic risk is elevated
        if panic_risk > 0.30:
            result["panic_warning"] = True
            result["max_position_size_pct"] *= 0.5  # Halve position sizes
            result["description"] += " [PANIC WARNING: elevated transition probability]"

        return result

    def _fallback_regime(self) -> dict[str, Any]:
        """Conservative fallback when HMM is not available or not fitted."""
        return {
            "regime": "LOW_VOL_TREND",
            "regime_id": 0,
            "panic_risk": 0.0,
            "transition_probs": {},
            "action_directive": self._get_action_directive(0, 0.0),
        }


# Singleton
hmm_classifier = HMMRegimeClassifier()
