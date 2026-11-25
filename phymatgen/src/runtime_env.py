"""Runtime environment setup for headless validation/repair jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import ctypes


def get_correction_env(name: str, default=None):
    """Read a correction setting, with the former prefix kept for compatibility."""
    return os.environ.get(
        f"PHYMATGEN_CORRECTION_{name}", os.environ.get(f"JIUZHENG_{name}", default)
    )


def ensure_conda_lib_on_loader_path() -> None:
    """Re-exec once so dlopen sees the active conda env lib directory first."""
    conda_lib_dir = Path(sys.prefix) / "lib"
    conda_libstdcxx = conda_lib_dir / "libstdc++.so.6"
    if (
        not conda_libstdcxx.exists()
        or os.environ.get("PHYMATGEN_CORRECTION_LIB_REEXEC") == "1"
        or (sys.argv and sys.argv[0] == "-c")
    ):
        return

    current_paths = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    conda_lib_str = str(conda_lib_dir)
    if current_paths and current_paths[0] == conda_lib_str:
        return

    env = dict(os.environ)
    env["PHYMATGEN_CORRECTION_LIB_REEXEC"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([conda_lib_str, *[p for p in current_paths if p != conda_lib_str]])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def configure_headless_runtime() -> None:
    """Set writable font/matplotlib config paths before heavy scientific imports."""
    conda_libstdcxx = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if conda_libstdcxx.exists():
        try:
            ctypes.CDLL(str(conda_libstdcxx), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    openbabel_libdir = Path(sys.prefix) / "lib" / "openbabel" / "3.1.0"
    if openbabel_libdir.is_dir():
        os.environ.setdefault("BABEL_LIBDIR", str(openbabel_libdir))

    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
    font_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", "/tmp"))
    try:
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        (font_cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("FONTCONFIG_PATH", "/tmp")
    if "FONTCONFIG_FILE" in os.environ:
        return

    fontconfig_file = Path("/tmp/phymatgen-correction-fonts.conf")
    try:
        if not fontconfig_file.exists():
            fontconfig_file.write_text(
                """<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/usr/share/fonts</dir>
  <dir>/usr/local/share/fonts</dir>
  <dir>~/.local/share/fonts</dir>
  <cachedir>/tmp/fontconfig</cachedir>
  <config></config>
</fontconfig>
""",
                encoding="utf-8",
            )
        os.environ["FONTCONFIG_FILE"] = str(fontconfig_file)
    except OSError:
        pass
