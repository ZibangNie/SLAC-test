from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def resolve_default_config_path() -> Path:
    return Path(__file__).resolve().parent / "pipeline_config.yaml"


def preparse_config_arg(argv: Optional[Sequence[str]] = None) -> Tuple[Optional[str], list[str]]:
    """
    Parse only --config first, so we can load YAML before full argparse parsing.
    """
    mini = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    mini.add_argument("--config")
    ns, remaining = mini.parse_known_args(argv)
    return getattr(ns, "config", None), list(remaining)


def _import_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError(
            "PyYAML is required for pipeline config loading. "
            "Install it with: pip install pyyaml"
        ) from e
    return yaml


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_pipeline_config(config_path: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load YAML config.
    If config_path is None, try default pipeline_config.yaml.
    If default file does not exist, return empty config.
    """
    path = Path(config_path).expanduser().resolve() if config_path else resolve_default_config_path()

    if not path.exists():
        if config_path is None:
            return {}, None
        raise FileNotFoundError(f"Config file does not exist: {path}")

    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a dict, got: {type(data).__name__}")

    return data, path


def get_runner_config(raw_cfg: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """
    Merge:
      common + <section_name>
    """
    common = raw_cfg.get("common", {}) or {}
    section = raw_cfg.get(section_name, {}) or {}

    if not isinstance(common, dict):
        raise ValueError("Config field 'common' must be a dict")
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be a dict")

    return deep_merge_dict(common, section)


def filter_allowed_keys(
    cfg: Dict[str, Any],
    allowed_keys: set[str],
) -> Tuple[Dict[str, Any], list[str]]:
    kept: Dict[str, Any] = {}
    unknown: list[str] = []

    for k, v in (cfg or {}).items():
        if k in allowed_keys:
            kept[k] = v
        else:
            unknown.append(k)

    return kept, sorted(unknown)


def merge_resolved_args(
    defaults: Dict[str, Any],
    config_values: Dict[str, Any],
    cli_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Final precedence:
      defaults < config < cli
    """
    out = copy.deepcopy(defaults)
    out.update(config_values or {})
    out.update(cli_values or {})
    return out


def namespace_from_dict(d: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**d)