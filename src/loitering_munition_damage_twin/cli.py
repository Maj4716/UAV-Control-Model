"""Stable console entry points for modules that retain script-style mains."""

import argparse
import os
import runpy
import sys
from typing import List

from loitering_munition_damage_twin.paths import PROJECT_ROOT


def _run_module(module: str, argv: List[str]) -> None:
    os.chdir(str(PROJECT_ROOT))
    sys.argv = [sys.argv[0], *argv]
    runpy.run_module(module, run_name="__main__")


def stage0_generate() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Stage-0 dataset. This is intentionally guarded "
            "because it can overwrite expensive local artifacts."
        )
    )
    parser.add_argument(
        "--confirm-generation",
        action="store_true",
        help="Confirm that a new Stage-0 generation run is explicitly intended.",
    )
    args = parser.parse_args()
    if not args.confirm_generation:
        parser.error("refusing to generate without --confirm-generation")
    _run_module(
        "loitering_munition_damage_twin.stage0.generation",
        [],
    )


def train() -> None:
    _run_module(
        "loitering_munition_damage_twin.surrogate.training",
        sys.argv[1:],
    )


def evaluate() -> None:
    _run_module(
        "loitering_munition_damage_twin.surrogate.evaluation",
        sys.argv[1:],
    )
