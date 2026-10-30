"""Command line entry point.

python -m imigraph compare    [config options]           # Brownian vs graph, print the report
python -m imigraph report     [--out output] [--label x] # ... and write a new HTML page
python -m imigraph gas        [config options]           # hard-disk gas, print the checks
python -m imigraph gas-report [--out output] [--label x] # ... and write a new HTML page
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from typing import Any

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.experiments.gas import GasConfig, run_gas


def add_config_options(parser: argparse.ArgumentParser, config_class: type) -> None:
    defaults = config_class()
    for field in fields(config_class):
        default = getattr(defaults, field.name)
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(default),
            default=default,
            help=f"default {default}",
        )


def add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", default="output", help="directory or .html file (default: output/)"
    )
    parser.add_argument(
        "--label", default="", help="short name of the change, part of the file name"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imigraph", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, config_class, help_text in (
        ("compare", ExperimentConfig, "Brownian motion vs lattice graph: print the comparison"),
        ("report", ExperimentConfig, "Brownian motion vs lattice graph: write the HTML page"),
        ("gas", GasConfig, "hard-disk gas: print the checks against kinetic theory"),
        ("gas-report", GasConfig, "hard-disk gas: write the HTML page"),
    ):
        sub = commands.add_parser(name, help=help_text)
        if name.endswith("report"):
            add_output_options(sub)
        add_config_options(sub, config_class)
    return parser


def config_from(args: argparse.Namespace, config_class: type) -> Any:
    return config_class(**{f.name: getattr(args, f.name) for f in fields(config_class)})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.startswith("gas"):
        result = run_gas(config_from(args, GasConfig))
        print(result.report.to_markdown())
        if args.command == "gas-report":
            from imigraph.report.gas import write_gas_report

            path = write_gas_report(args.out, result=result, label=args.label)
            print(f"\nWrote {path.resolve()}")
        return 0 if result.passed else 1
    result = run_experiment(config_from(args, ExperimentConfig))
    print(result.report.to_markdown())
    if args.command == "report":
        from imigraph.report import write_report

        path = write_report(args.out, result=result, label=args.label)
        print(f"\nWrote {path.resolve()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
