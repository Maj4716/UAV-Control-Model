#!/usr/bin/env python3
"""Measure the duration of one sim_engine DamageEngine evaluation."""

from __future__ import annotations

import argparse
import statistics
from time import perf_counter

from loitering_munition_damage_twin.simulation.engine import (
    DamageEngine,
    EncounterCondition,
    create_medium_loitering_munition,
    load_armor_plates,
    load_vehicle_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one representative sim_engine simulation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="number of measured simulations after warmup (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="number of unmeasured warmup simulations (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fragment spread RNG seed used by each simulation (default: 42)",
    )
    return parser.parse_args()


def benchmark(runs: int, warmup: int, seed: int) -> None:
    if runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if warmup < 0:
        raise ValueError("--warmup must be 0 or greater")

    setup_start = perf_counter()
    components = load_vehicle_model()
    armor_plates = load_armor_plates()
    projectile = create_medium_loitering_munition()
    encounter = EncounterCondition.from_speed_and_attitude(
        dz=200,
        pitch_deg=-90,
        speed=100,
    )
    engine = DamageEngine(armor_plates=armor_plates)
    setup_seconds = perf_counter() - setup_start

    for _ in range(warmup):
        engine.evaluate(projectile, encounter, components, rng_seed=seed)

    durations = []
    last_result = None
    for _ in range(runs):
        start = perf_counter()
        last_result = engine.evaluate(projectile, encounter, components, rng_seed=seed)
        durations.append(perf_counter() - start)

    mean_seconds = statistics.fmean(durations)
    median_seconds = statistics.median(durations)
    stdev_seconds = statistics.stdev(durations) if len(durations) > 1 else 0.0

    print("sim_engine single-simulation benchmark")
    print("case: medium front-detonation munition, dz=200 cm, pitch=-90 deg, speed=100 m/s")
    print(f"loaded components: {len(components)}")
    print(f"loaded armor plates: {len(armor_plates)}")
    print(f"setup time excluded from simulation timing: {setup_seconds * 1000:.3f} ms")
    print(f"warmup runs: {warmup}")
    print(f"measured runs: {runs}")
    print(f"representative single simulation (median): {median_seconds * 1000:.3f} ms")
    print(f"mean single simulation: {mean_seconds * 1000:.3f} ms")
    print(f"stdev: {stdev_seconds * 1000:.3f} ms")
    print(f"min: {min(durations) * 1000:.3f} ms")
    print(f"max: {max(durations) * 1000:.3f} ms")

    if last_result is not None:
        print(
            "last result: "
            f"fragments={last_result.total_fragments}, "
            f"hits={last_result.total_hits}, "
            f"penetrations={last_result.total_penetrations}, "
            f"damaged_components={last_result.damaged_count}"
        )


def main() -> None:
    args = parse_args()
    benchmark(args.runs, args.warmup, args.seed)


if __name__ == "__main__":
    main()
