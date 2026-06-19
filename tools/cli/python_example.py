import argparse
import importlib
import sys
from pathlib import Path


def _prefer_build_bin() -> None:
    build_bin = Path(__file__).resolve().parents[2] / "build" / "bin"
    if build_bin.is_dir():
        sys.path.insert(0, str(build_bin))


_prefer_build_bin()


llama_cpp_cli = importlib.import_module("llama_cpp_cli")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    return llama_cpp_cli.run([
        "-m", args.model,
        "--simple-io",
        "--single-turn",
        "-p", args.prompt,
    ])


if __name__ == "__main__":
    raise SystemExit(main())