"""Command line entry point.

python -m imigraph compare    [config options]           # Brownian vs graph, print the report
python -m imigraph report     [--out output] [--label x] # ... and write a new HTML page
python -m imigraph gas        [config options]           # hard disks vs lattice gas: print checks
python -m imigraph gas-report [--out output] [--label x] # ... and write a new HTML page

Lattice gas options carry the prefix ``--lattice-`` (``--lattice-size``, ``--lattice-boundary``).
The graph rule of ``compare`` / ``report`` can be given as JSON: ``--rule '{...}'`` (see
``imigraph.graph.rules.RuleSpec``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from typing import Any

from imigraph.experiments import ExperimentConfig, run_experiment
from imigraph.experiments.gas import GasConfig, LatticeGasConfig, run_gas_comparison

SCALARS = (int, float, str, bool)


def add_config_options(
    parser: argparse.ArgumentParser, config_class: type, prefix: str = ""
) -> None:
    defaults = config_class()
    for field in fields(config_class):
        default = getattr(defaults, field.name)
        name = f"--{prefix}{field.name.replace('_', '-')}"
        if field.name == "rule":
            parser.add_argument(name, type=json.loads, default=None, help="rule spec as JSON")
        elif isinstance(default, SCALARS) and not isinstance(default, bool):
            parser.add_argument(
                name, type=type(default), default=default, help=f"default {default}"
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
    for name, help_text in (
        ("compare", "Brownian motion vs lattice graph: print the comparison"),
        ("report", "Brownian motion vs lattice graph: write the HTML page"),
        ("gas", "hard-disk gas vs lattice gas: print the checks"),
        ("gas-report", "hard-disk gas vs lattice gas: write the HTML page"),
    ):
        sub = commands.add_parser(name, help=help_text)
        if name.endswith("report"):
            add_output_options(sub)
        if name.startswith("gas"):
            add_config_options(sub, GasConfig)
            add_config_options(sub, LatticeGasConfig, prefix="lattice-")
        else:
            add_config_options(sub, ExperimentConfig)
    return parser


def config_from(args: argparse.Namespace, config_class: type, prefix: str = "") -> Any:
    values = {}
    for field in fields(config_class):
        name = f"{prefix}{field.name}"
        if hasattr(args, name):
            values[field.name] = getattr(args, name)
    return config_class(**values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.startswith("gas"):
        result = run_gas_comparison(
            config_from(args, GasConfig), config_from(args, LatticeGasConfig, prefix="lattice_")
        )
        print(result.gas.report.to_markdown())
        print()
        print(result.lattice.report.to_markdown())
        print()
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
