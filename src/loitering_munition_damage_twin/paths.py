"""Repository-local paths shared by command-line entry points."""

from pathlib import Path


def _discover_project_root() -> Path:
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").is_file():
        return source_candidate

    cwd_candidate = Path.cwd().resolve()
    if (cwd_candidate / "pyproject.toml").is_file():
        return cwd_candidate

    return source_candidate


PROJECT_ROOT = _discover_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output"
ABLATION_CONFIG_ROOT = PROJECT_ROOT / "configs" / "ablations"
EXPERIMENT_OUTPUT_ROOT = OUTPUT_ROOT / "experiments"
