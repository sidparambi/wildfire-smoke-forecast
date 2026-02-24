"""Map PM2.5 → 0-100 Smoke Advisory Score using EPA AQI breakpoints."""

from __future__ import annotations

# EPA AQI PM2.5 breakpoints (μg/m³) and corresponding advisory score ranges
# (low_pm25, high_pm25, low_score, high_score, tier_name, color)
_BREAKPOINTS = [
    (0.0, 12.0, 0, 20, "Low", "green"),
    (12.0, 35.4, 20, 50, "Moderate", "yellow"),
    (35.4, 55.4, 50, 75, "High", "orange"),
    (55.4, 150.4, 75, 100, "Hazardous", "red"),
    (150.4, float("inf"), 100, 100, "Hazardous", "red"),
]


def pm25_to_advisory(pm25: float) -> dict:
    """Convert a PM2.5 concentration to an advisory score and risk tier.

    Args:
        pm25: PM2.5 concentration in μg/m³ (non-negative)

    Returns:
        dict with keys:
            advisory_score  int [0, 100]
            risk_tier       str  Low | Moderate | High | Hazardous
            color           str  green | yellow | orange | red
    """
    pm25 = max(0.0, float(pm25))

    for low_c, high_c, low_s, high_s, tier, color in _BREAKPOINTS:
        if pm25 <= high_c:
            if high_c == low_c or high_s == low_s:
                score = high_s
            else:
                # Linear interpolation within tier
                score = low_s + (pm25 - low_c) / (high_c - low_c) * (high_s - low_s)
            return {
                "advisory_score": int(round(min(100, max(0, score)))),
                "risk_tier": tier,
                "color": color,
            }

    # Fallback (should not reach here)
    return {"advisory_score": 100, "risk_tier": "Hazardous", "color": "red"}


def confidence_interval(pm25_pred: float, model_rmse: float = 8.0) -> tuple[float, float]:
    """Return a ±1σ confidence interval around the PM2.5 prediction.

    Uses model_rmse as a proxy for prediction uncertainty.
    """
    lower = max(0.0, round(pm25_pred - model_rmse, 2))
    upper = round(pm25_pred + model_rmse, 2)
    return lower, upper
