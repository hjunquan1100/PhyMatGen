"""Runtime environment setup for training and evaluation entry points."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path


def ensure_conda_lib_on_loader_path() -> None:
    conda_lib_dir = Path(sys.prefix) / "lib"
    conda_libstdcxx = conda_lib_dir / "libstdc++.so.6"
    if not conda_libstdcxx.exists() or os.environ.get("PHYMATGEN_LIB_REEXEC") == "1":
        return

    current_paths = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    conda_lib_str = str(conda_lib_dir)
    if current_paths and current_paths[0] == conda_lib_str:
        return

    env = dict(os.environ)
    env["PHYMATGEN_LIB_REEXEC"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(
        [conda_lib_str, *[path for path in current_paths if path != conda_lib_str]]
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def configure_runtime() -> None:
    _patch_invalid_networkx_backends()

    conda_lib_dir = Path(sys.prefix) / "lib"
    conda_libstdcxx = conda_lib_dir / "libstdc++.so.6"
    if conda_libstdcxx.exists():
        try:
            ctypes.CDLL(str(conda_libstdcxx), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    openbabel_libdir = conda_lib_dir / "openbabel" / "3.1.0"
    if openbabel_libdir.is_dir():
        os.environ.setdefault("BABEL_LIBDIR", str(openbabel_libdir))

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


def _patch_invalid_networkx_backends() -> None:
    for module_name in ("importlib.metadata", "importlib_metadata"):
        try:
            metadata_module = importlib.import_module(module_name)
        except ImportError:
            continue

        original = getattr(metadata_module, "entry_points", None)
        entry_points_type = getattr(metadata_module, "EntryPoints", None)
        if original is None or getattr(original, "_phymatgen_networkx_patch", False):
            continue

        def wrapped_entry_points(
            *args,
            _original=original,
            _entry_points_type=entry_points_type,
            **kwargs,
        ):
            entry_points = _original(*args, **kwargs)
            if isinstance(entry_points, dict):
                return entry_points
            try:
                filtered = [
                    entry_point
                    for entry_point in entry_points
                    if not (
                        getattr(entry_point, "group", None) == "networkx.backends"
                        and not str(getattr(entry_point, "name", "")).isidentifier()
                    )
                ]
            except TypeError:
                return entry_points
            if entry_points_type is None:
                return filtered
            try:
                return entry_points_type(filtered)
            except Exception:
                return filtered

        wrapped_entry_points._phymatgen_networkx_patch = True
        metadata_module.entry_points = wrapped_entry_points

    for key in list(os.environ):
        if not key.upper().startswith("NETWORKX_"):
            continue
        values = [value.strip() for value in os.environ[key].split(",")]
        valid_values = [value for value in values if value and value.isidentifier()]
        if valid_values:
            os.environ[key] = ",".join(valid_values)
        else:
            del os.environ[key]
