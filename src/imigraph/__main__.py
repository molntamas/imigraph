"""Command line entry point: ``python -m imigraph report [--out output] [--seed N] ...``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields

from imigraph.experiments import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imigraph", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_config_options(subparser: argparse.ArgumentParser) -> None:
        defaults = ExperimentConfig()
        for field in fields(ExperimentConfig):
            subparser.add_argument(
                f"--{field.name.replace('_', '-')}",
                type=field.type
                if field.type in (int, float)
                else type(getattr(defaults, field.name)),
                default=getattr(defaults, field.name),
                help=f"default {getattr(defaults, field.name)}",
            )

    compare = commands.add_parser("compare", help="run the experiment and print the report")
    add_config_options(compare)

    report = commands.add_parser("report", help="run the experiment and write the HTML page")
    report.add_argument(
        "--out", default="output", help="directory or .html file (default: output/)"
    )
    add_config_options(report)
    return parser


def config_from(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(**{f.name: getattr(args, f.name) for f in fields(ExperimentConfig)})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from(args)
    result = run_experiment(config)
    print(result.report.to_markdown())
    if args.command == "report":
        from imigraph.report import write_report

        path = write_report(args.out, result=result)
        print(f"\nWrote {path.resolve()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
