"""
format_utils.py — Shared number-formatting helpers.

Several tabs had their own near-identical "format this as K/M" inline
logic (volume, OBV, force index, etc.) that had quietly drifted apart —
different decimal places, `>` vs `>=` thresholds, some places rounding to
1 decimal, others to 2. None of it was wrong exactly, just inconsistent in
a way a user would notice scanning between tabs. Centralizing the common
cases here so there's one definition to keep consistent instead of N that
slowly diverge.

This does not attempt to replace every one-off format string in the
codebase — plenty of `f"{v:.2f}"` calls are fine as-is. It targets the
formatters that were duplicated across multiple files/functions, which is
where drift actually shows up to a user.
"""


def fmt_compact(value, decimals: int = 1) -> str:
    """Compact K/M notation for large magnitudes (volume, OBV, force index,
    etc.). Uses consistent >= thresholds and decimal places everywhere it's
    called, instead of each call site picking its own.

        fmt_compact(1_234_567)   -> "1.2M"
        fmt_compact(45_000)      -> "45.0K"
        fmt_compact(212)         -> "212.0"
    """
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.{decimals}f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def fmt_currency(value, decimals: int = 2, fallback: str = "—") -> str:
    """`$1,234.56`-style currency formatting with thousands separators."""
    if value is None:
        return fallback
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def fmt_pct(value, decimals: int = 2, signed: bool = True, fallback: str = "—") -> str:
    """`+1.23%` / `-4.56%`-style percentage formatting."""
    if value is None:
        return fallback
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    sign = "+" if signed else ""
    return f"{v:{sign}.{decimals}f}%"
