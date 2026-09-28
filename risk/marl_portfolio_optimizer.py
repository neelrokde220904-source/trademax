"""Multi-Agent Reinforcement Learning (MARL) + Hierarchical Risk Parity (HRP) Portfolio Optimizer.

Replaces static Kelly Criterion with three competing RL agents + structural diversification.
Falls back to fractional Kelly when RL agents lack sufficient training data (< 252 days).

Required: stable-baselines3, gymnasium, scipy, numpy, torch
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from config.settings import settings

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("stable-baselines3 / gymnasium not installed. MARL optimizer will use HRP-only fallback.")


class PortfolioAllocationEnv(gym.Env):
    """Custom RL environment for portfolio allocation.

    Observation: [returns, volatility, momentum, regime_features] per asset
    Action: Continuous allocation weights for each asset
    Reward: Configured per agent (Sortino, Information Ratio, etc.)
    """

    def __init__(self, price_data: np.ndarray, n_assets: int,
                 reward_type: str = "sortino", lookback: int = 60) -> None:
        super().__init__()
        self.price_data = price_data
        self.n_assets = n_assets
        self.lookback = lookback
        self.reward_type = reward_type
        self.current_step = lookback
        self.max_steps = len(price_data) - 1

        # Observation: (lookback returns, vol, momentum) per asset
        obs_dim = n_assets * 3  # returns, volatility, momentum per asset
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        # Action: allocation weight per asset (softmax applied externally)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.portfolio_value = 1.0
        self.returns_history = []
        return self._get_obs(), {}

    def step(self, action):
        # Normalize action to weights (softmax)
        weights = np.exp(action) / np.sum(np.exp(action))

        # Calculate portfolio return
        if self.current_step < len(self.price_data) - 1:
            asset_returns = (
                self.price_data[self.current_step + 1] / self.price_data[self.current_step]
            ) - 1.0
            portfolio_return = np.dot(weights, asset_returns)
        else:
            portfolio_return = 0.0

        self.portfolio_value *= (1 + portfolio_return)
        self.returns_history.append(portfolio_return)
        self.current_step += 1

        # Reward
        reward = self._calculate_reward(portfolio_return)
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        start = max(0, self.current_step - self.lookback)
        window = self.price_data[start:self.current_step + 1]

        obs = []
        for asset_idx in range(self.n_assets):
            asset_prices = window[:, asset_idx]
            if len(asset_prices) < 2:
                obs.extend([0.0, 0.0, 0.0])
                continue
            returns = np.diff(np.log(asset_prices + 1e-8))
            obs.append(float(np.mean(returns)))  # Mean return
            obs.append(float(np.std(returns)))    # Volatility
            obs.append(float(asset_prices[-1] / asset_prices[0] - 1))  # Momentum
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, portfolio_return: float) -> float:
        if self.reward_type == "sortino":
            if len(self.returns_history) < 10:
                return portfolio_return
            downside = [r for r in self.returns_history[-60:] if r < 0]
            downside_std = np.std(downside) if downside else 1e-8
            return float(np.mean(self.returns_history[-60:]) / max(downside_std, 1e-8))
        elif self.reward_type == "information_ratio":
            if len(self.returns_history) < 10:
                return portfolio_return
            # Benchmark: equal weight
            excess = portfolio_return - 0.0  # Simplified
            tracking_error = np.std(self.returns_history[-60:]) if len(self.returns_history) >= 60 else 1e-8
            return float(excess / max(tracking_error, 1e-8))
        return portfolio_return


class HierarchicalRiskParity:
    """Groups assets by correlation, then allocates risk inversely.

    Prevents sector concentration (e.g., 5 banking stocks dominating portfolio).
    """

    def compute_weights(self, symbols: list[str], price_matrix: np.ndarray) -> dict[str, float]:
        """Compute HRP portfolio weights.

        Args:
            symbols: List of asset symbols.
            price_matrix: Price matrix of shape (n_days, n_assets).

        Returns:
            {symbol: weight} dictionary.
        """
        n_assets = len(symbols)
        if n_assets == 0:
            return {}
        if n_assets == 1:
            return {symbols[0]: 1.0}

        # Log returns
        returns = np.diff(np.log(price_matrix + 1e-8), axis=0)

        if returns.shape[0] < 2:
            # Equal weight fallback
            w = 1.0 / n_assets
            return {s: w for s in symbols}

        # Correlation matrix
        corr_matrix = np.corrcoef(returns.T)
        # Handle numerical issues
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method="single")

        # Quasi-diagonalize
        sorted_indices = self._quasi_diagonalize(linkage_matrix, n_assets)

        # Recursive bisection
        weights = self._recursive_bisection(returns, sorted_indices)

        return {symbols[i]: float(weights[i]) for i in range(n_assets)}

    def _quasi_diagonalize(self, linkage_mat: np.ndarray, n: int) -> list[int]:
        """Sort assets by hierarchical cluster order."""
        from scipy.cluster.hierarchy import leaves_list
        return list(leaves_list(linkage_mat))

    def _recursive_bisection(self, returns: np.ndarray, sorted_indices: list[int]) -> np.ndarray:
        """Bisect the tree and allocate inversely proportional to cluster variance."""
        n = len(sorted_indices)
        weights = np.ones(returns.shape[1])

        cluster_items = [sorted_indices]

        while cluster_items:
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Variance of each sub-cluster
                left_var = self._cluster_variance(returns, left)
                right_var = self._cluster_variance(returns, right)

                # Allocate inversely proportional to variance
                total_var = left_var + right_var
                if total_var > 0:
                    alloc_left = 1.0 - left_var / total_var
                    alloc_right = 1.0 - alloc_left
                else:
                    alloc_left = alloc_right = 0.5

                for idx in left:
                    weights[idx] *= alloc_left
                for idx in right:
                    weights[idx] *= alloc_right

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            cluster_items = new_clusters

        # Normalize
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def _cluster_variance(self, returns: np.ndarray, indices: list[int]) -> float:
        """Calculate variance of an equally-weighted sub-portfolio."""
        if not indices:
            return 0.0
        sub_returns = returns[:, indices]
        portfolio_returns = np.mean(sub_returns, axis=1)
        return float(np.var(portfolio_returns))


class MARLPortfolioOptimizer:
    """Three RL agents compete to allocate capital.

    Each uses different objectives → diverse, robust portfolio.
    Falls back to HRP-only when RL data is insufficient.
    """

    MIN_TRAINING_DAYS = 252  # 1 year of data before RL is reliable

    def __init__(self) -> None:
        self.hrp = HierarchicalRiskParity()
        self.ppo_agent = None
        self.sac_agent = None
        self.rl_trained = False

    def train_rl_agents(self, price_matrix: np.ndarray, n_assets: int,
                        total_timesteps: int = 50_000) -> None:
        """Train PPO and SAC agents on historical price data.

        Run this OFFLINE (not during market hours).
        """
        if not RL_AVAILABLE:
            logger.warning("RL libraries not available. Skipping RL training.")
            return

        if price_matrix.shape[0] < self.MIN_TRAINING_DAYS:
            logger.warning(
                "Insufficient data for RL training: {} days (need {})",
                price_matrix.shape[0], self.MIN_TRAINING_DAYS,
            )
            return

        logger.info("Training PPO agent (Sortino objective)...")
        ppo_env = PortfolioAllocationEnv(price_matrix, n_assets, reward_type="sortino")
        self.ppo_agent = PPO("MlpPolicy", ppo_env, verbose=0,
                             learning_rate=3e-4, n_steps=2048, batch_size=64)
        self.ppo_agent.learn(total_timesteps=total_timesteps)

        logger.info("Training SAC agent (Information Ratio objective)...")
        sac_env = PortfolioAllocationEnv(price_matrix, n_assets, reward_type="information_ratio")
        self.sac_agent = SAC("MlpPolicy", sac_env, verbose=0,
                             learning_rate=3e-4, batch_size=256)
        self.sac_agent.learn(total_timesteps=total_timesteps)

        self.rl_trained = True
        logger.info("RL agents trained successfully")

    def optimize_portfolio(
        self,
        signals: dict[str, Any],
        prices: dict[str, np.ndarray],
        regime: dict[str, Any],
        available_capital: float,
    ) -> dict[str, float]:
        """Compute optimized portfolio weights.

        Returns: {symbol: allocation_pct}

        Strategy:
        - HRP always runs (structural diversification)
        - RL agents added when trained and regime allows
        - Meta-aggregation weights shift toward HRP during stress
        """
        # Filter: only symbols with BUY signals
        allowed_symbols = [
            s for s in signals
            if signals[s].get("action") == "BUY" or signals[s].get("signal") in ("BUY", "STRONG_BUY")
        ]

        if not allowed_symbols:
            return {}

        # Build price matrix for allowed symbols
        min_len = min(len(prices.get(s, [])) for s in allowed_symbols if s in prices)
        if min_len < 20:
            # Not enough data, equal weight
            w = 1.0 / len(allowed_symbols)
            return {s: w for s in allowed_symbols}

        price_matrix = np.column_stack([
            np.array(prices[s][-min_len:]) for s in allowed_symbols if s in prices
        ])

        # Step 1: HRP weights (always compute)
        hrp_weights = self.hrp.compute_weights(allowed_symbols, price_matrix)

        # Step 2: RL weights (if trained and enough data)
        ppo_weights = {}
        sac_weights = {}
        if self.rl_trained and RL_AVAILABLE and self.ppo_agent and self.sac_agent:
            ppo_weights = self._predict_rl_weights(self.ppo_agent, allowed_symbols, price_matrix)
            sac_weights = self._predict_rl_weights(self.sac_agent, allowed_symbols, price_matrix)

        # Step 3: Meta-aggregation
        panic_risk = regime.get("panic_risk", 0.0)
        hrp_meta_weight = 0.4 + (panic_risk * 0.4)  # HRP gets more weight during stress
        rl_weight = 1.0 - hrp_meta_weight

        final_weights: dict[str, float] = {}
        for symbol in allowed_symbols:
            w = hrp_meta_weight * hrp_weights.get(symbol, 0)
            if ppo_weights and sac_weights:
                w += (rl_weight * 0.5) * ppo_weights.get(symbol, 0)
                w += (rl_weight * 0.5) * sac_weights.get(symbol, 0)
            else:
                w += rl_weight * hrp_weights.get(symbol, 0)  # HRP fallback for RL
            final_weights[symbol] = w

        return self._normalize_weights(final_weights, max_single=0.12)

    def _predict_rl_weights(self, agent: Any, symbols: list[str],
                            price_matrix: np.ndarray) -> dict[str, float]:
        """Get RL agent predictions for current state."""
        try:
            n_assets = len(symbols)
            env = PortfolioAllocationEnv(price_matrix, n_assets)
            obs, _ = env.reset()
            action, _ = agent.predict(obs, deterministic=True)
            # Softmax
            weights = np.exp(action) / np.sum(np.exp(action))
            return {symbols[i]: float(weights[i]) for i in range(n_assets)}
        except Exception as e:
            logger.warning("RL prediction failed: {}. Using uniform.", e)
            w = 1.0 / len(symbols)
            return {s: w for s in symbols}

    @staticmethod
    def _normalize_weights(weights: dict[str, float], max_single: float = 0.12) -> dict[str, float]:
        """Normalize weights to sum to 1.0 and cap any single position."""
        total = sum(weights.values())
        if total <= 0:
            return weights

        normalized = {s: w / total for s, w in weights.items()}

        # Iteratively cap and redistribute
        for _ in range(10):
            excess = 0.0
            uncapped = 0
            for s, w in normalized.items():
                if w > max_single:
                    excess += w - max_single
                    normalized[s] = max_single
                else:
                    uncapped += 1
            if excess <= 0 or uncapped <= 0:
                break
            per_uncapped = excess / uncapped
            for s in normalized:
                if normalized[s] < max_single:
                    normalized[s] += per_uncapped

        # Final normalize
        total = sum(normalized.values())
        if total > 0:
            normalized = {s: w / total for s, w in normalized.items()}

        return normalized


# Singleton
marl_optimizer = MARLPortfolioOptimizer()
