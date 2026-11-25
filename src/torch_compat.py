"""Compatibility helpers for environments with mismatched PyTorch Dynamo files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def ensure_torch_dynamo_compat() -> None:
    dynamo_dir = Path(torch.__file__).resolve().parent / "_dynamo"
    if not dynamo_dir.exists() or "torch._dynamo.utils" in sys.modules:
        return

    try:
        dynamo_pkg = type(sys)("torch._dynamo")
        dynamo_pkg.__path__ = [str(dynamo_dir)]
        dynamo_pkg.__package__ = "torch._dynamo"
        sys.modules["torch._dynamo"] = dynamo_pkg

        for submodule in ("metrics_context", "graph_utils"):
            source = dynamo_dir / f"{submodule}.py"
            module_name = f"torch._dynamo.{submodule}"
            if not source.exists() or module_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            module.__package__ = "torch._dynamo"
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            setattr(dynamo_pkg, submodule, module)

        utils_source = dynamo_dir / "utils.py"
        if not utils_source.exists():
            return
        utils_spec = importlib.util.spec_from_file_location("torch._dynamo.utils", utils_source)
        if utils_spec is None or utils_spec.loader is None:
            return
        utils_module = importlib.util.module_from_spec(utils_spec)
        utils_module.__package__ = "torch._dynamo"
        sys.modules["torch._dynamo.utils"] = utils_module
        utils_spec.loader.exec_module(utils_module)
        if not hasattr(utils_module, "NP_SUPPORTED_MODULES"):
            utils_module.NP_SUPPORTED_MODULES = frozenset()

        sys.modules.pop("torch._dynamo", None)
        preloaded_utils = sys.modules.get("torch._dynamo.utils")

        class DynamoInjectFinder:
            def find_spec(self, fullname, path, target=None):
                if fullname != "torch._dynamo":
                    return None
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is None or spec.loader is None:
                        continue
                    original_exec = spec.loader.exec_module

                    def exec_with_utils(module):
                        if preloaded_utils is not None:
                            setattr(module, "utils", preloaded_utils)
                        original_exec(module)

                    spec.loader.exec_module = exec_with_utils
                    return spec
                return None

        sys.meta_path.insert(0, DynamoInjectFinder())
    except Exception:
        for module_name in list(sys.modules):
            if module_name.startswith("torch._dynamo"):
                sys.modules.pop(module_name, None)
