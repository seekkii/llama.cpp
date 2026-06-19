from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import ctypes
from pathlib import Path
from types import ModuleType


def _extension_suffixes() -> list[str]:
    return list(getattr(importlib.machinery, "EXTENSION_SUFFIXES", []))


def _candidate_paths(base_dir: Path) -> list[Path]:
    suffixes = _extension_suffixes()
    candidates: list[Path] = []

    search_dirs = [
        base_dir.parent.parent / "build" / "tools" / "cli",
        base_dir.parent.parent / "build" / "bin",
        base_dir.parent.parent / "build-cli-python-test" / "tools" / "cli",
        base_dir,
    ]
    search_dirs.extend(sorted(base_dir.parent.parent.glob("build*/tools/cli")))
    search_dirs.extend(sorted(base_dir.parent.parent.glob("build*/bin")))

    seen: set[Path] = set()
    for search_dir in search_dirs:
        if not search_dir.is_dir() or search_dir in seen:
            continue
        seen.add(search_dir)

        for suffix in suffixes:
            exact = search_dir / f"llama_cpp_cli{suffix}"
            if exact.exists():
                candidates.append(exact)

        for candidate in sorted(search_dir.glob("llama_cpp_cli*.so")):
            candidates.append(candidate)

    return candidates


def _copy_to_exec_path(source_path: Path) -> Path:
    digest = hashlib.sha256(str(source_path.resolve()).encode("utf-8")).hexdigest()[:12]
    target_dir = Path(tempfile.gettempdir()) / "llama_cpp_cli" / digest
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name

    needs_copy = True
    if target_path.exists():
        try:
            needs_copy = source_path.stat().st_mtime_ns != target_path.stat().st_mtime_ns
        except OSError:
            needs_copy = True

    if needs_copy:
        shutil.copy2(source_path, target_path)
        target_path.chmod(0o755)

    return target_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_build_bin_dirs() -> list[Path]:
    repo_root = _repo_root()
    candidates = [repo_root / "build" / "bin"]
    candidates.extend(sorted(repo_root.glob("build*/bin")))
    return [path for path in candidates if path.is_dir()]


def _parse_ldd(binary_path: Path) -> list[Path]:
    result = subprocess.run(
        ["ldd", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    dependencies: list[Path] = []
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].strip().split("(", 1)[0].strip()
        if resolved.startswith("/"):
            dependencies.append(Path(resolved))

    return dependencies


def _stage_local_dependencies(staging_dir: Path, module_path: Path) -> list[Path]:
    build_bin_dirs = [path.resolve() for path in _local_build_bin_dirs()]
    staged: list[Path] = []

    for dep_path in _parse_ldd(module_path):
        resolved = dep_path.resolve()
        if not any(str(resolved).startswith(str(build_dir) + os.sep) or resolved == build_dir for build_dir in build_bin_dirs):
            continue

        staged_path = staging_dir / dep_path.name
        shutil.copy2(dep_path, staged_path)
        staged_path.chmod(0o755)
        staged.append(staged_path)

    return staged


def _preload_libraries(library_paths: list[Path]) -> None:
    pending = list(dict.fromkeys(library_paths))
    mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)

    while pending:
        next_pending: list[Path] = []
        progress = False

        for library_path in pending:
            try:
                ctypes.CDLL(str(library_path), mode=mode)
                progress = True
            except OSError:
                next_pending.append(library_path)

        if not progress:
            break

        pending = next_pending


def _load_from_path(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("llama_cpp_cli", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create a module spec for '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["llama_cpp_cli"] = module
    spec.loader.exec_module(module)
    return module


def load_module() -> ModuleType:
    base_dir = Path(__file__).resolve().parent
    candidates = _candidate_paths(base_dir)
    if not candidates:
        raise ImportError(
            "Could not find `llama_cpp_cli` extension. Build it first with CMake or the provided compile script."
        )

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _load_from_path(candidate)
        except ImportError as exc:
            last_error = exc
            if "failed to map segment from shared object" not in str(exc):
                continue

            copied_path = _copy_to_exec_path(candidate)
            staged_libraries = _stage_local_dependencies(copied_path.parent, copied_path)
            _preload_libraries(staged_libraries)
            return _load_from_path(copied_path)

    if last_error is not None:
        raise last_error

    raise ImportError("Failed to load `llama_cpp_cli` for an unknown reason.")


llama_cpp_cli = load_module()
run = llama_cpp_cli.run
run_script = llama_cpp_cli.run_script
