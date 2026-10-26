"""Command line: `gold-vol fetch` and `gold-vol evaluate`."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .data import DEFAULT_CACHE, EXCHANGE_TZ, SYMBOL, update_cache


def fetch(cache: Path) -> None:
    bars, stats = update_cache(cache)
    days = bars.index.tz_convert(EXCHANGE_TZ).normalize().nunique()
    print(
        f"{SYMBOL}: +{stats.added:,} new bars, {stats.revised:,} revised. "
        f"Cache now holds {stats.total:,} one-minute bars, "
        f"{bars.index[0]:%Y-%m-%d %H:%M} to {bars.index[-1]:%Y-%m-%d %H:%M} UTC "
        f"({days} trading days). Latest close: ${bars['close'].iloc[-1]:,.2f}."
    )
    print(f"Saved to {cache}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gold-vol",
        description="Forecast next-hour volatility of COMEX gold futures.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"minute-bar cache (default: {DEFAULT_CACHE})",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download new bars and extend the cache")
    evaluate = commands.add_parser(
        "evaluate", help="run the walk-forward evaluation and write reports"
    )
    evaluate.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="output folder (default: reports)",
    )

    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch(args.cache)
    else:
        # Imported here so `fetch` doesn't load scikit-learn and matplotlib.
        from .evaluation import run

        run(args.cache, args.reports)
