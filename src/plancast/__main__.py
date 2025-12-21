"""Command-line entry point: ``python -m plancast run [--config ...] [--out ...] [--fast]``."""

from __future__ import annotations

import argparse

from .data import ROOT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="plancast", description="Forecasting financial line items under macro scenarios.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline and write reports/")
    r.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    r.add_argument("--out", default=str(ROOT / "reports"))
    r.add_argument("--fast", action="store_true", help="shorter evaluation window and fewer paths (for CI)")
    args = ap.parse_args(argv)
    from .pipeline import run

    run(args.config, args.out, fast=args.fast)
    print(f"Reports written to {args.out}")


if __name__ == "__main__":
    main()
