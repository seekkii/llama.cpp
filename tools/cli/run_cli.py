import argparse
import sys
from pathlib import Path


def _prefer_build_bin() -> None:
    build_bin = Path(__file__).resolve().parents[2] / "build" / "bin"
    if build_bin.is_dir():
        sys.path.insert(0, str(build_bin))


def _load_module():
    _prefer_build_bin()
    try:
        import llama_cpp_cli
        return llama_cpp_cli
    except ImportError:
        from llama_cpp_cli_loader import llama_cpp_cli
        return llama_cpp_cli


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run llama.cpp's cli.cpp entrypoint through the Python nanobind module.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to llama-cli. Prefix with '--' to stop argparse parsing.",
    )
    parsed = parser.parse_args()

    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args.pop(0)

    if not forwarded_args:
        parser.error("no llama-cli arguments provided; pass them after '--'")

    module = _load_module()
    return module.run(forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())