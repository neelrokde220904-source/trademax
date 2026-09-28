"""Optuna Hyperparameter Optimizer — walk-forward validated strategy optimization."""

from __future__ import annotations

from typing import Any, Callable

import optuna
from loguru import logger

from backtester.engine import BacktestEngine
from backtester.metrics import calculate_metrics
from config.settings import settings

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Parameter Search Spaces ────────────────────────────

PARAM_SPACES: dict[str, dict[str, tuple]] = {
    "MomentumBreakout": {
        "volume_multiplier": (1.2, 3.5, "float"),
        "rsi_low": (45, 60, "int"),
        "rsi_high": (65, 80, "int"),
        "trailing_atr_mult": (1.5, 5.0, "float"),
        "rr_ratio": (1.0, 4.0, "float"),
    },
    "SupertrendADX": {
        "supertrend_length": (5, 20, "int"),
        "supertrend_multiplier": (1.5, 5.0, "float"),
        "adx_threshold": (15, 40, "int"),
        "rr_ratio": (1.0, 3.0, "float"),
    },
}


def _sample_params(trial: optuna.Trial, strategy_name: str) -> dict[str, Any]:
    """Sample hyperparameters from the search space for a given strategy."""
    space = PARAM_SPACES.get(strategy_name, {})
    params: dict[str, Any] = {}
    for name, (low, high, ptype) in space.items():
        if ptype == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif ptype == "float":
            params[name] = trial.suggest_float(name, low, high, step=0.1)
    return params


def _make_strategy_fn(strategy_name: str, params: dict[str, Any]) -> Callable:
    """Create a strategy function with the given parameters."""
    if strategy_name == "MomentumBreakout":
        from strategies.momentum_breakout import MomentumBreakout
        strategy = MomentumBreakout(params=params)

        def strategy_fn(symbol, df, bar_index):
            if bar_index < strategy.lookback_52w:
                return []
            signals = strategy._analyse(symbol, df)
            return [signals] if signals else []

        return strategy_fn

    elif strategy_name == "SupertrendADX":
        from strategies.supertrend_adx import SupertrendADX
        strategy = SupertrendADX(params=params)

        def strategy_fn(symbol, df, bar_index):
            if bar_index < 30:
                return []
            signals = strategy._analyse(symbol, df)
            return [signals] if signals else []

        return strategy_fn

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def optimize_strategy(
    strategy_name: str,
    data: dict[str, Any],
    n_trials: int = 50,
    objective_metric: str = "sharpe_ratio",
    initial_capital: float | None = None,
) -> dict[str, Any]:
    """Optimize a strategy's hyperparameters using Optuna.

    Args:
        strategy_name: "MomentumBreakout" or "SupertrendADX".
        data: {symbol: DataFrame} training data.
        n_trials: Number of optimization trials.
        objective_metric: Metric to maximize ("sharpe_ratio", "net_pnl", "calmar_ratio", "profit_factor").
        initial_capital: Starting capital (defaults to settings).

    Returns:
        Dict with best_params, best_value, all trial results.
    """
    capital = initial_capital or settings.initial_capital

    if strategy_name not in PARAM_SPACES:
        logger.error("[Optimizer] Unknown strategy: {}", strategy_name)
        return {"best_params": {}, "best_value": float("-inf")}

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, strategy_name)
        strategy_fn = _make_strategy_fn(strategy_name, params)

        engine = BacktestEngine(initial_capital=capital)
        result = engine.run(data, strategy_fn)
        metrics = calculate_metrics(result)

        # Primary metric
        value = getattr(metrics, objective_metric, 0.0)

        # Penalize strategies with too few trades (statistically unreliable)
        if metrics.total_trades < 5:
            value = float("-inf")

        # Penalize excessive drawdown
        if metrics.max_drawdown_pct > 25:
            value *= 0.5

        return value if value == value else float("-inf")  # NaN guard

    study = optuna.create_study(
        direction="maximize",
        study_name=f"optimize_{strategy_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    logger.info(
        "[Optimizer] Starting {} trials for {} (metric: {})",
        n_trials, strategy_name, objective_metric,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(
        "[Optimizer] Best {} = {:.4f} with params: {}",
        objective_metric, best.value, best.params,
    )

    return {
        "strategy": strategy_name,
        "best_params": best.params,
        "best_value": best.value,
        "best_trial_number": best.number,
        "n_trials": n_trials,
        "objective_metric": objective_metric,
        "trials_summary": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None and t.value != float("-inf")
        ][:20],  # Top 20 by trial number
    }


def create_optimized_strategy_fn(strategy_name: str, train_data: dict[str, Any]) -> Callable:
    """Optimize and return a strategy function — compatible with walk-forward's optimize_fn.

    Usage with walk-forward:
        engine.run_walk_forward(
            data, default_strategy_fn,
            optimize_fn=lambda train: create_optimized_strategy_fn("SupertrendADX", train),
        )
    """
    result = optimize_strategy(strategy_name, train_data, n_trials=30)
    best_params = result.get("best_params", {})
    return _make_strategy_fn(strategy_name, best_params)
