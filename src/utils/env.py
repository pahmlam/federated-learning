"""Environment helpers for local, per-machine FL configuration."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

_PREFIX = "FL_"
_ENV_TO_FIELD = {
    "FL_RUN_ID": "exp_id",
    "FL_OUTPUT_DIR": "output_dir",
    "FL_MANIFEST_PATH": "manifest_path",
    "FL_DATA_ROOT": "root_dir",
    "FL_CLIENT_ID": "client_id",
    "FL_NUM_CLIENTS": "num_clients",
    "FL_NUM_CLASSES": "num_classes",
    "FL_IMAGE_SIZE": "image_size",
    "FL_BATCH_SIZE": "batch_size",
    "FL_LOCAL_EPOCHS": "local_epochs",
    "FL_CENTRALIZED_EPOCHS": "centralized_epochs",
    "FL_NUM_ROUNDS": "num_rounds",
    "FL_LR": "lr",
    "FL_MOMENTUM": "momentum",
    "FL_WEIGHT_DECAY": "weight_decay",
    "FL_NUM_WORKERS": "num_workers",
    "FL_SCORE_THRESHOLD": "score_threshold",
    "FL_DEVICE": "device",
    "FL_PRETRAINED": "pretrained",
    "FL_SEED": "seed",
    "FL_MIN_TRAIN_NODES": "min_train_nodes",
    "FL_MIN_EVALUATE_NODES": "min_evaluate_nodes",
    "FL_MIN_AVAILABLE_NODES": "min_available_nodes",
    "FL_EDGE_PROFILE": "edge_profile",
    "FL_EDGE_PROFILES": "edge_profiles",
}
_LEGACY_ENV_TO_FIELD = {
    "FL_DET_EXP_ID": "exp_id",
    "FL_DET_OUTPUT_DIR": "output_dir",
    "FL_DET_MANIFEST_PATH": "manifest_path",
    "FL_DET_ROOT_DIR": "root_dir",
    "FL_DET_CLIENT_ID": "client_id",
    "FL_DET_NUM_CLIENTS": "num_clients",
    "FL_DET_NUM_CLASSES": "num_classes",
    "FL_DET_IMAGE_SIZE": "image_size",
    "FL_DET_BATCH_SIZE": "batch_size",
    "FL_DET_LOCAL_EPOCHS": "local_epochs",
    "FL_DET_CENTRALIZED_EPOCHS": "centralized_epochs",
    "FL_DET_NUM_ROUNDS": "num_rounds",
    "FL_DET_LR": "lr",
    "FL_DET_MOMENTUM": "momentum",
    "FL_DET_WEIGHT_DECAY": "weight_decay",
    "FL_DET_NUM_WORKERS": "num_workers",
    "FL_DET_SCORE_THRESHOLD": "score_threshold",
    "FL_DET_DEVICE": "device",
    "FL_DET_PRETRAINED": "pretrained",
    "FL_DET_SEED": "seed",
}
_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load ``.env`` values without requiring python-dotenv at test time."""

    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    file_values: dict[str, str] = {}
    if env_path.is_file():
        try:
            from dotenv import dotenv_values

            file_values = {
                str(key): str(value)
                for key, value in dotenv_values(env_path).items()
                if key is not None and value is not None
            }
        except ImportError:
            file_values = _parse_env_file(env_path)

    # Match python-dotenv's default behavior: existing process env wins.
    merged = dict(file_values)
    for key, value in os.environ.items():
        if key.startswith(_PREFIX):
            merged[key] = value
    return merged


def fl_env_values(
    config_cls: type,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return parsed ``FL_*`` values keyed by a config dataclass field name.

    ``FL_DET_*`` names are compatibility aliases. When both namespaces map to
    the same field, the system-level ``FL_*`` name wins.
    """

    field_types = {field.name: field.type for field in fields(config_cls)}
    selected: dict[str, tuple[str, str]] = {}
    raw_values = load_env_file(path)

    for env_name, raw_value in raw_values.items():
        if raw_value == "" or env_name not in _LEGACY_ENV_TO_FIELD:
            continue
        field_name = _LEGACY_ENV_TO_FIELD[env_name]
        if field_name in field_types:
            selected[field_name] = (env_name, raw_value)

    for env_name, raw_value in raw_values.items():
        if raw_value == "" or env_name not in _ENV_TO_FIELD:
            continue
        field_name = _ENV_TO_FIELD[env_name]
        if field_name in field_types:
            selected[field_name] = (env_name, raw_value)

    return {
        field_name: _parse_value(raw_value, field_types[field_name], env_name)
        for field_name, (env_name, raw_value) in selected.items()
    }


def detection_env_values(
    config_cls: type,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return parsed FL env values for the current detection workload."""

    return fl_env_values(config_cls, path=path)


def fl_client_id_from_env(*, path: str | Path | None = None) -> str | None:
    raw_values = load_env_file(path)
    raw_value = raw_values.get("FL_CLIENT_ID", raw_values.get("FL_DET_CLIENT_ID"))
    if raw_value is None or raw_value == "":
        return None
    return raw_value


def detection_client_id_from_env(*, path: str | Path | None = None) -> str | None:
    return fl_client_id_from_env(path=path)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _parse_value(raw_value: str, annotation: Any, env_name: str) -> Any:
    target = _resolve_annotation(annotation)
    if target is bool:
        return _parse_bool(raw_value, env_name)
    try:
        if target is int:
            return int(raw_value)
        if target is float:
            return float(raw_value)
        if target is str:
            return raw_value
    except ValueError as exc:
        raise ValueError(f"Invalid value for {env_name}: {raw_value!r}") from exc
    return raw_value


def _parse_bool(raw_value: str, env_name: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    choices = ", ".join(sorted(_BOOL_TRUE | _BOOL_FALSE))
    raise ValueError(
        f"Invalid boolean for {env_name}: {raw_value!r}. Expected one of {choices}"
    )


def _resolve_annotation(annotation: Any) -> Any:
    if isinstance(annotation, str):
        # ``from __future__ import annotations`` makes field types strings, e.g.
        # "int" or "int | None". Strip an optional ``| None`` before resolving so
        # nullable numeric fields parse as their base type instead of falling to str.
        parts = [part for part in annotation.replace(" ", "").split("|") if part != "None"]
        key = parts[0] if len(parts) == 1 else annotation
        return {"str": str, "int": int, "float": float, "bool": bool}.get(key, str)
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args[0] if len(args) == 1 else str
